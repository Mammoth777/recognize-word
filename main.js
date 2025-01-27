import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from './data'; // 假设你将数据加载逻辑放在 data.js 中

let breakTrain = false
document.getElementById('break').addEventListener('click', () => {
  breakTrain = true
})

const filenames = [
  'ds_aa',
  'ds_ab',
  'ds_ac',
  'ds_ad',
  'ds_ae',
  'ds_af',
  'ds_ag',
  'ds_ah',
  'ds_ai',
  'ds_aj',
  'ds_ak',
  'ds_al',
  'ds_am',
  'ds_an',
  'ds_ao',
  'ds_ap',
  'ds_aq',
  'ds_ar',
  'ds_as',
  'ds_at',
  'ds_au',
  'ds_av',
  'ds_aw',
  'ds_ax',
  'ds_ay',
  'ds_az',
  'ds_ba',
  'ds_bb',
  'ds_bc',
  'ds_bd',
  'ds_be',
  'ds_bf',
  'ds_bg',
  'ds_bh',
  'ds_bi',
  'ds_bj',
  'ds_bk',
  'ds_bl',
  'ds_bm',
  'ds_bn',
  'ds_bo',
  'ds_bp',
  'ds_bq',
  'ds_br',
  'ds_bs',
  'ds_bt',
  'ds_bu',
  'ds_bv',
  'ds_bw',
  'ds_bx',
  'ds_by',
  'ds_bz',
  'ds_ca',
  'ds_cb',
  'ds_cc',
  'ds_cd',
  'ds_ce',
  'ds_cf',
  'ds_cg',
  'ds_ch',
  'ds_ci',
  'ds_cj',
  'ds_ck',
  'ds_cl',
  'ds_cm',
  'ds_cn',
  'ds_co',
  'ds_cp',
  'ds_cq',
  'ds_cr',
  'ds_cs',
  'ds_ct',
  'ds_cu',
  'ds_cv',
  'ds_cw',
  'ds_cx',
  'ds_cy',
  'ds_cz',
  'ds_da',
  'ds_db',
  'ds_dc',
  'ds_dd',
  'ds_de',
  'ds_df',
  'ds_dg',
  'ds_dh',
  'ds_di',
  'ds_dj',
  'ds_dk',
  'ds_dl',
  'ds_dm',
  'ds_dn',
  'ds_do',
  'ds_dp',
  'ds_dq',
  'ds_dr',
  'ds_ds',
  'ds_dt',
  'ds_du',
  'ds_dv',
  'ds_dw',
  'ds_dx',
  'ds_dy',
  'ds_dz',
  'ds_ea',
  'ds_eb',
  'ds_ec',
  'ds_ed',
  'ds_ee',
  'ds_ef',
  'ds_eg',
  'ds_eh',
  'ds_ei',
  'ds_ej',
  'ds_ek',
  'ds_el',
  'ds_em',
  'ds_en',
  'ds_eo',
  'ds_ep',
  'ds_eq',
  'ds_er',
  'ds_es',
  'ds_et',
  'ds_eu',
]
// tf.util.shuffle(filenames);

// 创建模型
function createModel() {
  const model = tf.sequential();

  // 第一层：卷积层
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 3,
    filters: 8,
    activation: 'relu'
  }));

  // 第二层：最大池化层
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // 第三层：卷积层
  model.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));

  // 第四层：最大池化层
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // 第五层：展平层
  model.add(tf.layers.flatten());

  // 第六层：全连接层
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));

  // 输出层：全连接层，10个输出类别（0-9）  改成26个字母试试
  model.add(tf.layers.dense({ units: 26, activation: 'softmax' }));

  return model;
}

// 编译模型
function compileModel(model) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
}

// 训练模型
async function trainModel(model, filenames) {
  const batchSize = 256;
  const trainEpochs = 3;

  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = { name: 'Model Training', tab: 'Training' };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  const callbacks = {
    ...fitCallbacks,
  }

  const loadData = async (index) => {
    const filename = filenames[index];
    console.log(`${index}. Loading data from ${filename}`);
    const data = new MnistData();
    await data.load(`/dataset/archive/${filename}`);
    return await data.getTrainData();
  }

  for (let i = 0; i < filenames.length; i++) {
    if (breakTrain) {
      break
    }
    const trainData = await loadData(i);
    await model.fit(trainData.images, trainData.labels, {
      batchSize,
      epochs: trainEpochs,
      validationSplit: 0.1,
      shuffle: true,
      callbacks
    });
    tf.dispose(trainData.images);
    tf.dispose(trainData.labels);
  }
}

// 评估模型
async function evaluateModel(model) {
  const filename = filenames[Math.floor(Math.random() * filenames.length)];
  console.log('Evaluating model with test data from', filename);
  const data = new MnistData();
  await data.load(`/dataset/archive/${filename}`);
  const testData = await data.getTestData();
  const evalOutput = model.evaluate(testData.images, testData.labels);

  console.log(`\nEvaluation result:
        Loss = ${evalOutput[0].dataSync()[0].toFixed(4)}
        Accuracy = ${evalOutput[1].dataSync()[0].toFixed(4)}`);
}

// 主函数
async function run() {
  const model = createModel();
  compileModel(model);
  tfvis.show.modelSummary({ name: 'Model Architecture' }, model);

  await trainModel(model, filenames);

  console.log('=== 训练后模型结构 ===');
  model.summary();
  // 3. 保存前验证权重
  const denseLayer = model.layers[model.layers.length - 1];
  console.log('输出层权重形状:', denseLayer.weights.map(w => w.shape));

  // 保存模型到本地文件
  await model.save('downloads://my-model-a-z');
  await evaluateModel(model);
}

async function predict(model, testData, index) {
  const testImage = testData.images.slice([index, 0, 0, 0], [1, 28, 28, 1]);
  const prediction = model.predict(testImage);
  let predictedLabel = prediction.argMax(1).dataSync();
  const image = testImage.div(255.0)
  const container = document.createElement('div')
  const canvas = document.createElement('canvas');
  canvas.id = `image-${index}`;
  const label = document.createElement('span');
  console.log(predictedLabel, 'predictedLabel')
  label.innerText = `${String.fromCharCode(predictedLabel[0] + 65)}`;
  container.appendChild(canvas);
  container.appendChild(label);
  const app = document.getElementById('app');
  app.appendChild(container);
  await tf.browser.toPixels(image.reshape([28, 28, 1]), document.getElementById(`image-${index}`));
}

async function batchPredict(modelPath, testDataPath) {
  const model = await tf.loadLayersModel(modelPath);
  const data = new MnistData();
  await data.load(testDataPath)
  const testData = await data.getTestData(testDataPath);
  const sampleCount = testData.images.shape[0];
  const spotcheckCount = 20;
  const startIndex = Math.floor(Math.random() * (sampleCount - spotcheckCount));
  for (let i = startIndex; i < startIndex + spotcheckCount; i++) {
    predict(model, testData, i);
  }
}

document.getElementById('run').addEventListener('click', run);
document.getElementById('predict').addEventListener('click', () => {
  // batchPredict('/model/my-model-1.json');
  const filename = filenames[Math.floor(Math.random() * filenames.length)]
  batchPredict('/model/my-model-a-z.json', `/dataset/archive/${filename}`);
});

// 让我看看第一行是个啥
async function showRowImg() {
  const ds = [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,64,64,99,170,64,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,128,128,146,255,255,255,255,255,227,66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,113,198,255,255,255,255,255,255,255,255,255,224,50,0,0,0,0,0,0,0,0,0,0,0,0,35,78,255,255,248,191,191,21,0,0,47,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,38,198,255,255,179,113,0,0,0,0,0,142,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,7,210,255,255,191,26,0,0,0,0,0,0,236,255,255,255,255,255,0,0,0,0,0,0,0,0,0,33,212,255,220,177,0,0,0,0,0,0,0,0,236,255,255,255,255,208,0,0,0,0,0,0,0,0,0,132,255,255,113,0,0,0,0,0,0,0,0,0,236,255,255,255,255,66,0,0,0,0,0,0,0,0,50,224,255,198,28,0,0,0,0,0,0,0,0,0,94,255,255,255,163,17,0,0,0,0,0,0,0,0,66,255,255,179,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,161,255,255,179,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,179,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,208,255,255,255,113,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,194,255,255,184,128,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,212,255,255,255,234,191,170,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,191,227,255,255,255,255,255,255,255,255,128,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,71,128,170,255,255,255,255,255,255,255,208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,64,64,64,64,234,255,128,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  const image = ds.slice(1)
  const imageTensor = tf.tensor(image, [28, 28, 1])
  tf.browser.toPixels(imageTensor.div(255.0), document.getElementById('demo'))
  const model = await tf.loadLayersModel('/model/my-model-a-z.json');
  // 1. 获取预测结果
  const prediction = model.predict(imageTensor.reshape([1, 28, 28, 1]))

  // 2. 显示所有类别的概率分布
  const probabilities = await prediction.data();
  console.log('预测概率分布:', probabilities);

  // 3. 获取最高概率的类别
  const predictedLabel = prediction.argMax(1).dataSync()[0];
  console.log('预测标签索引:', predictedLabel);
  console.log('预测字母:', String.fromCharCode(predictedLabel + 65));
  // 4. 显示前3个最可能的预测
  const topK = 3;
  const probs = Array.from(probabilities);
  const top3 = probs
    .map((prob, i) => ({prob, letter: String.fromCharCode(i + 65)}))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, topK);
    
  console.log('前3个预测:', top3);
  const span = document.createElement('span')
  span.innerText = String.fromCharCode(predictedLabel + 65)
  document.body.appendChild(span)

}

showRowImg()
