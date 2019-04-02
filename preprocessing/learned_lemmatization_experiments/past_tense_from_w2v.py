# %%
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
import utils

embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim50/1")
past_tense = [["afford", "afforded"], ["arrive", "arrived"], ["beam", "beamed"], ["bolt", "bolted"], ["camp", "camped"],
              ["crawl", "crawled"], ["dislike", "disliked"], ["hum", "hummed"], ["judge", "judged"],
              ["perform", "performed"], ["punish", "punished"], ["settle", "settled"], ["sniff", "sniffed"],
              ["tow", "towed"], ["tremble", "tremble"], ["wreck", "wrecked"]]

past_tense_flat = utils.flatten(past_tense)

embeddings = embed(past_tense_flat)

if __name__=="__main__":
    sess = tf.Session()
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    evaluated_embeddings = sess.run(embeddings)

    print(evaluated_embeddings)

    for i in range(len(past_tense)):
        diff = evaluated_embeddings[2*i+1] - evaluated_embeddings[2*i]
        plt.plot(diff, alpha=0.2)
    plt.show()
