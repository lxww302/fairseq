import numpy as np
import torch
import sentencepiece as spm
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from omegaconf import DictConfig
from langid import LangId


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
        tgt_lang = input_str[: 2].lower()
        src_text = input_str[2:].strip()
        src_lang = self.lang_id(src_text)['lang_code']
        tgt_text = self._translate(src_lang, tgt_lang, src_text)
        result = {'src_lang': src_lang,
                  'src_text': src_text,
                  'tgt_lang': tgt_lang,
                  'tgt_text': tgt_text}
        print(result)
        return tgt_text


def main():
    print('begin parsing')
    parser = options.get_generation_parser()
    print('parser ready')
    args = options.parse_args_and_arch(parser)
    print('args ready')
    cfg = convert_namespace_to_omegaconf(args)
    print('cfg ready')
    server = MultilingualTranslationServer(cfg)
    print(server._translate('en', 'zh', 'hello'))
    print(server.translate('en 我是一个粉刷匠，粉刷本领强。'))


if __name__ == '__main__':
    main()
