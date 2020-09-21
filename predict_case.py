from bert.cn_punctor import *


if __name__ == "__main__":
    flags.DEFINE_bool("do_train", False, "Whether to run training.")
    flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
    flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
    main()