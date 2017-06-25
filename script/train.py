import tensorflow as tf
import subprocess
import logging


MAX_STEPS = 300000
STEPS_PER_VALIDATION = 1000
STEPS_PER_CHECKPOINT = 20000
TEST_THRESHOLD = 200000


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')

    try:
        # Returns a Tensor with the contents of the given variable in the checkpoint.
        # Doc: https://www.tensorflow.org/api_docs/python/tf/contrib/framework/load_variable
        global_step = tf.contrib.framework.load_variable(checkpoint_dir="model", name="global_step")
    except:
        global_step = 0

    logging.info("Training starts with global_step=%d", global_step)

    while global_step < MAX_STEPS:
        terminate_step = max(global_step + STEPS_PER_CHECKPOINT, TEST_THRESHOLD)

        logging.info("Train from %d to %d", global_step, terminate_step)

        proc = ["python3", "src/summarization.py",
                "--max_iter", str(terminate_step),
                "--steps_per_validation", str(STEPS_PER_VALIDATION),
                "--steps_per_checkpoint", str(STEPS_PER_CHECKPOINT)]

        subprocess.call(proc)

        global_step = terminate_step

        subprocess.call(["python3", "script/test.py"])
