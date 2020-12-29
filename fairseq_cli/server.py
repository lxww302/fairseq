import numpy as np
import sentencepiece as spm
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from flask import Flask, jsonify, request
from langid import LangId
from omegaconf import DictConfig
from open_lark import OpenLark

app = Flask(__name__)
app_id = 'cli_9f0ce4677771500c'
app_secret = 'ZSzR44EGfB3l1kWZluuFt8Ic4KqwmTss'
encrypt_key = None
verification_token = 'EcpG00O5Ee7HEOsZN3cvAdJOj4nGGxtF'


class MultilingualTranslationServer(object):
    def __init__(self, cfg: DictConfig) -> None:
        cfg.dataset.max_tokens = 12000
        if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
            np.random.seed(cfg.common.seed)
            utils.set_torch_seed(cfg.common.seed)
        print('setup_task')
        task = tasks.setup_task(cfg.task)
        use_cuda = torch.cuda.is_available() and not cfg.common.cpu
        print(
            "loading model(s) from {}".format(cfg.common_eval.path))
        models, _model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides={},
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
        print('prepare_for_inference_')
        for model in models:
            if model is None:
                continue
            if cfg.common.fp16:
                model.half()
            if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)
        extra_gen_cls_kwargs = {"lm_model": None,
                                "lm_weight": cfg.generation.lm_weight}
        print('build_generator')
        generator = task.build_generator(
            models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('/mnt/nlp-aws/wuwei.ai/models/m2m_100/spm.128k.model')
        self.cfg = cfg
        self.task = task
        self.models = models
        self.use_cuda = use_cuda
        self.generator = generator
        self.dict = self.task.source_dictionary
        self.lang_id = LangId()

    def _translate(self, src_lang: str, tgt_lang: str, src_text: str):
        self.task.args.target_lang = tgt_lang
        self.task.dicts[tgt_lang] = self.dict

        src_lang_id = self.dict.symbols.index('__' + src_lang + '__')
        tgt_lang_id = self.dict.symbols.index('__' + tgt_lang + '__')
        print('src_lang_id', src_lang_id)

        src_text_tokens = self.sp.EncodeAsPieces(src_text)
        print('src_text_tokens', src_text_tokens)
        src_text_ids = [src_lang_id] + \
            [self.dict.symbols.index(token) for token in src_text_tokens] + [2]
        print('src_text_ids', src_text_ids)
        sample = {'net_input': {'src_tokens': torch.tensor([src_text_ids]),
                                'src_lengths': torch.tensor([len(src_text_ids)])}}
        if self.use_cuda:
            sample = utils.move_to_cuda(sample)
        print('sample', sample)
        hypos = self.task.inference_step(self.generator, self.models, sample)
        print('hypos', hypos)
        hypo_tokens = hypos[0][0]['tokens'].cpu().int()
        print('hypo_tokens', hypo_tokens)
        hypo_str = self.dict.string(
            hypo_tokens,
            self.cfg.common_eval.post_process,
            escape_unk=True,
            extra_symbols_to_ignore={2, tgt_lang_id}
        )
        return hypo_str

    def translate(self, input_str):
        input_str = input_str.strip()
        tgt_lang = input_str[: 2].lower()
        src_text = input_str[2:].strip()
        src_lang = self.lang_id(src_text)['lang_code']
        if tgt_lang not in self.task.langs:
            tgt_lang = 'en'
        if src_lang not in self.task.langs:
            src_lang = 'zh'
        tgt_text = self._translate(src_lang, tgt_lang, src_text)
        result = {'src_lang': src_lang,
                  'src_text': src_text,
                  'tgt_lang': tgt_lang,
                  'tgt_text': tgt_text}
        print(result)
        return tgt_text


parser = options.get_generation_parser()
args = options.parse_args_and_arch(parser)
cfg = convert_namespace_to_omegaconf(args)
server = MultilingualTranslationServer(cfg)


def handle_message(msg_uuid, msg_timestamp, event, json_event):
    print(msg_uuid, msg_timestamp, json_event, "ssss")
    open_message_id = json_event["open_message_id"]
    open_chat_id = json_event["open_chat_id"]
    ret_dict = {"open_message_id": open_message_id,
                "open_chat_id": open_chat_id,
                "reply": server.translate(json_event["text_without_at_bot"])}

    return ret_dict


@app.route("/bot", methods=["GET", "POST"])
def index():
    body = request.get_json()
    bot = OpenLark(app_id=app_id,
                   app_secret=app_secret,
                   encrypt_key=encrypt_key,
                   oauth_redirect_uri="10.100.196.196:5555",
                   verification_token=verification_token)
    ret = bot.handle_callback(body, handle_message)
    if isinstance(ret, dict) and ("challenge" in ret):
        print('1'*100)
        return jsonify(ret)
    else:
        print('0'*100)
        if "open_message_id" in ret:
            open_message_id = ret["open_message_id"]
            open_chat_id = ret["open_chat_id"]
            msgid = bot.reply(open_message_id).to_open_chat_id(
                open_chat_id).send_text(ret["reply"])
            # bot.urgent_message(msgid, [zsj_open_id], urgent_type=UrgentType.sms)
        return jsonify({"ok": True})


if __name__ == '__main__':
    print(app.before_first_request_funcs)
    app.run(host='0.0.0.0', port='5555')
