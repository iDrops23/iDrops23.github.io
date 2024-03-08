import { env, AutoProcessor, AutoModel, RawImage } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.15.1';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;

// Reference the elements that we will need
const status = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const example = document.getElementById('example');

const EXAMPLE_URL = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/city-streets.jpg';

const THRESHOLD = 0.25;

// Create a new object detection pipeline
status.textContent = 'Loading model...';
const processor = await AutoProcessor.from_pretrained('Xenova/yolov9-c_all');

// For this demo, we resize the image so that its shortest edge is 256px
processor.feature_extractor.size = { shortest_edge: 256 }

const model = await AutoModel.from_pretrained('Xenova/yolov9-c_all');
status.textContent = 'Ready';

example.addEventListener('click', (e) => {
    e.preventDefault();
    detect(EXAMPLE_URL);
});

fileUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();

    // Set up a callback when the file is loaded
    reader.onload = e2 => detect(e2.target.result);

    reader.readAsDataURL(file);
});


// Detect objects in the image
async function detect(url) {
    // Update UI
    imageContainer.innerHTML = '';

    // Read image
    const image = await RawImage.fromURL(url);

    // Set container width and height depending on the image aspect ratio
    const ar = image.width / image.height;
    const [cw, ch] = (ar > 1) ? [640, 640 / ar] : [640 * ar, 640];
    imageContainer.style.width = `${cw}px`;
    imageContainer.style.height = `${ch}px`;
    imageContainer.style.backgroundImage = `url(${url})`;

    status.textContent = 'Analysing...';

    // Preprocess image
    const inputs = await processor(image);

    // Predict bounding boxes
    const { outputs } = await model(inputs);

    status.textContent = '';

    const sizes = inputs.reshaped_input_sizes[0].reverse();
    outputs.tolist().forEach(x => renderBox(x, sizes));
}

// Render a bounding box and label on the image
function renderBox([xmin, ymin, xmax, ymax, score, id], [w, h]) {
    if (score < THRESHOLD) return; // Skip boxes with low confidence

    // Generate a random color for the box
    const color = '#' + Math.floor(Math.random() * 0xFFFFFF).toString(16).padStart(6, 0);

    // Draw the box
    const boxElement = document.createElement('div');
    boxElement.className = 'bounding-box';
    Object.assign(boxElement.style, {
        borderColor: color,
        left: 100 * xmin / w + '%',
        top: 100 * ymin / h + '%',
        width: 100 * (xmax - xmin) / w + '%',
        height: 100 * (ymax - ymin) / h + '%',
    })

    // Draw label
    const labelElement = document.createElement('span');
    labelElement.textContent = model.config.id2label[id];
    labelElement.className = 'bounding-box-label';
    labelElement.style.backgroundColor = color;

    boxElement.appendChild(labelElement);
    imageContainer.appendChild(boxElement);
}
