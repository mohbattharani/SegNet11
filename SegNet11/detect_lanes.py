import tensorflow as tf
from modelUpdated import *

def get_road_pixels(dense_pred, im_rgb):
    output_image = dense_pred.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, -1)
    x = np.argmax(output_image,axis=2)
    #print ('Shape of x', x.shape)
    #im = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH,3), dtype=np.uint8)
    for i,_ in enumerate(x):
        for j,_ in enumerate(x[i]):
            value = x[i][j]
            if (value != 3):
                im_rgb[i][j] = [0,0,0]
    return im_rgb


def test():
  test_data_node = tf.placeholder(tf.float32,
        shape=[TEST_BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
  test_labels_node = tf.placeholder(tf.int64, shape=[TEST_BATCH_SIZE, 360, 480, 1])
  phase_train = tf.placeholder(tf.bool, name='phase_train')
  loss, logits = inference(test_data_node, test_labels_node, TEST_BATCH_SIZE, phase_train)

  pred = tf.argmax(logits, axis=3)
  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()
  saver = tf.train.Saver(variables_to_restore)

  with tf.Session() as sess:
    # Load checkpoint
    try:
      print("Trying to restore last checkpoint from: ",path_ckpt)
      last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=path_ckpt)
      print ('last chkr point:', last_chk_path)
    # Try and load the data in the checkpoint.
      saver.restore(sess, save_path=last_chk_path)
    # If we get to this point, the checkpoint was successfully loaded.
      print("Restored checkpoint from:", last_chk_path)
    except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
      print("Failed to restore checkpoint. Initializing variables instead.")
      sess.run(tf.global_variables_initializer())

    threads = tf.train.start_queue_runners(sess=sess)
    i = 2
    #im_filename = '/home/mohbat/RoadSegmentation/road lane Seg/SegNet11/road/Seq05VD_f00510.png'
    im_filename = '/home/mohbat/RoadSegmentation/DataSet/CamSeq01/test/0016E5_07965.png'
    im_rgb = np.array(skimage.io.imread(im_filename), np.float32)
    image_batch = im_rgb[np.newaxis]     # converts into list
    feed_dict = {
        test_data_node: image_batch,
        phase_train: False
    }
    dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
    im_rgb_road = get_road_pixels(dense_prediction, im_rgb)
      # output_image to verify
    if (True):
        writeImage(im[0], 'road/pred_image'+str(i)+'.png')
        scp.misc.imsave('road/road_rgb'+str(i)+'.png', im_rgb_road)
        print ('Result saved.')
          #writeImage(dense_prediction, 'pred_image.png')


def main(args):
    #checkArgs()
    test()
    #model.training()

if __name__ == '__main__':
  tf.app.run()
