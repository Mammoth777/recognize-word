import * as tf from '@tensorflow/tfjs';

// const trainData = '/dataset/archive/xaa'
// const testData = '/dataset/archive/xaa'

export class MnistData {
  constructor() {
    this.dataset = null;
  }

  async load(trainData) {
    this.dataset = await tf.data.csv(trainData, {
      columnNames: ['label', ...Array.from({ length: 784 }, (_, i) => 'px' + i.toString())],
    });
  }

  async getTrainData() {
    const images = [];
    const labels = [];

    await this.dataset.forEachAsync((row) => {
      const label = row['label'];
      const image = Object.values(row).slice(1); // 提取像素值
      labels.push(label);
      images.push(image);
    });
    return {
      images: tf.tensor2d(images, [images.length, 784]).reshape([-1, 28, 28, 1]),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), 26)
    };
  }

  async getTestData() {
    const images = []
    const labels = []
    await this.dataset.forEachAsync((row) => {
      const label = row['label']
      const image = Object.values(row).slice(1)
      labels.push(label)
      images.push(image)
    })
    return {
      images: tf.tensor2d(images, [images.length, 784]).reshape([-1, 28, 28, 1]),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), 26)
    }
  }
}