import codecs
import pickle
import os
import tensorflow as tf
import numpy as np
from bert.cn_punctor import flags, filed_based_convert_examples_to_features, FLAGS, \
    file_based_input_fn_builder, PunctorProcessor, softmax_model_fn_builder
from bert import tokenization, modeling
flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_predict", False,
                  "Whether to run eval on the dev set.")
# flags.DEFINE_string('data_format', 'value', 'The explanation of this parameter is ing')
# flags.DEFINE_bool("do_infer", True, "Whether to run eval on the dev set.")
bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

tpu_cluster_resolver = None
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=FLAGS.master,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.num_tpu_cores,
        per_host_input_for_training=is_per_host))

num_train_steps = None
num_warmup_steps = None

processors = {
    'punctor': PunctorProcessor,
}
task_name = FLAGS.task_name.lower()
processor = processors[task_name]()
label_list = processor.get_labels()
tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}
model_fn = softmax_model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_list),
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_tpu)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=FLAGS.train_batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    predict_batch_size=FLAGS.predict_batch_size)


def infer(input_file_path):
    predict_examples = processor.get_infer_examples(FLAGS.test_data_dir, input_file_path)

    predict_file = os.path.join(FLAGS.output_dir, "infer.tf_record")
    filed_based_convert_examples_to_features(
        predict_examples, label_list, FLAGS.max_seq_length, tokenizer, predict_file, mode="test")

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    if FLAGS.use_tpu:
        # Warning: According to tpu_estimator.py Prediction on TPU is an
        # experimental feature and hence not supported here
        raise ValueError("Prediction in TPU not supported")
    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    # print(result)
    final_results = []
    for predict_line, prediction in zip(predict_examples, result):
        prediction = np.argmax(prediction, axis=-1)
        line = []
        line_token = str(predict_line.text).split(' ')
        print(line_token)
        print(prediction)
        for word, label in zip(line_token, prediction):
            curr_labels = id2label[label]
            if curr_labels == "_SPACE":
                line.append(word)
            else:
                line.append(word)
                line.append(curr_labels)
        line = ' ' + ' '.join(line)
        print(line)
        final_results.append(line)
    return final_results


if __name__ == "__main__":

    infer('/root/Projects/PunctuationPrediction-BERT/data/raw/LREC/case.txt')
    input_file_path = '/root/Projects/PunctuationPrediction-BERT/data/raw/LREC/case2.txt'
    print("-----------------")
    open(input_file_path, "w").write("我 是 谁")
    infer(input_file_path)
