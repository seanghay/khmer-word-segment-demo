const $loader = document.querySelector('#loader');
const $app = document.querySelector('#app');


(async function () {

  /**
   * @type {import('@tensorflow/tfjs-tflite')}
   */
  const tflite = window.tflite
  const model = await tflite.loadTFLiteModel('https://cdn.jsdelivr.net/gh/Socret360/akara-android@main/akara/src/main/assets/khmer_word_seg_model.tflite')
  
  $app.style.display = 'block';
  $loader.style.display = 'none';


  const POS_MAP = 'AB;AUX;CC;CD;DT;IN;JJ;VB;NN;PN;PA;PRO;QT;RB;SYM;NS'.split(';');
  const CHAR_MAP = "!#%&()*+,-./0123456789<=>?@[\\]^_កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ឴឵ាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័្។៕ៗ៘៙៚៛០១២៣៤៥៦៧៨៩​"

  const N_UNIQUE_CHARS = 133
  const N_UNIQUE_POS = 16
  const MAX_SENTENCE_LENGTH = 687

  async function segment(text) {
    return new Promise(async resolve => {
      setTimeout(async () => {
        const inputVector = create2dArray(MAX_SENTENCE_LENGTH, N_UNIQUE_CHARS)
  
        let i = 0;
  
        for (const char of text) {
          let charIndex = CHAR_MAP.indexOf(char);
  
          if (charIndex == -1) {
            charIndex = CHAR_MAP[CHAR_MAP.length - 1]
          }
  
          inputVector[i][charIndex] = 1.0;
          i++;
        }
  
        const result = model.predict(tf.tensor([inputVector]))
        const [predicted] = await result.array();
  
        let tmp = '';
        let j = 0;
  
        const wordBreaks = [];
  
        for (const char of text) {
          const posVec = predicted[j];
          let posIndex = posVec.indexOf(Math.max(...posVec));
          if (posIndex == -1) posIndex = 15
          const pos = POS_MAP[posIndex];
  
          if (pos === "NS") {
            tmp += char;
          } else {
            if (tmp.length > 0) {
              wordBreaks.push(tmp);
              tmp = "";
            }
            tmp += char;
          }
  
          j++
        }
  
        if (tmp.length > 0) {
          wordBreaks.push(tmp)
        }
  
        resolve(wordBreaks)
      })
    })
  }
  
  
  const $output = document.querySelector('#output')
  
  document.querySelector('#input').addEventListener('input', debounce(async (e) => {
    const { value } = e.target;
    const segments = await segment(value);
    $output.value = segments.join(' ')
  }, 200))

})()



function create2dArray(x, y) {
  return new Array(x).fill(0.0).map(() => new Array(y).fill(0.0))
}



function debounce(func, wait, immediate) {
  var timeout;
  return function () {
    var context = this, args = arguments;
    var later = function () {
      timeout = null;
      if (!immediate) func.apply(context, args);
    };
    var callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    if (callNow) func.apply(context, args);
  };
};