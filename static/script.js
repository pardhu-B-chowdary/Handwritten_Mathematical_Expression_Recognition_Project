// --- Global State ---
let activeSource = 'none'; // 'upload' or 'canvas'
let abortController = null;
let cropper = null;
let isUpdatingCrop = false;

// --- Socket.IO Room Logic ---
const socket = io();

const urlParams = new URLSearchParams(window.location.search);
let roomId = urlParams.get('room');

if (!roomId) {
    roomId = Math.random().toString(36).substring(2, 10);
    window.history.replaceState({}, '', '?room=' + roomId);
}

socket.on('connect', () => {
    socket.emit('join', { room: roomId });
});

// Backward compatibility or fallback
socket.on('sync_image', (data) => {
    const imagePreviewContainer = document.getElementById("imagePreviewContainer");
    const imagePreview = document.getElementById("imagePreview");
    if (imagePreview) {
        imagePreview.setAttribute("src", data.image);
        imagePreviewContainer.style.display = "block";
        activeSource = 'upload';
        
        if (cropper) cropper.destroy();
        setTimeout(initCropper, 100);
    }
});

socket.on('sync_prediction', (data) => {
    const latexBox = document.getElementById("latexBox");
    const mathOutput = document.getElementById("mathOutput");
    
    if (latexBox) latexBox.value = data.latex;
    if (mathOutput) mathOutput.innerHTML = data.latex.trim() ? "$$ " + data.latex + " $$" : "";
    
    if (window.MathJax) {
        MathJax.typesetPromise();
    }
});

function initCropper() {
    const imagePreview = document.getElementById("imagePreview");
    if (cropper) {
        cropper.destroy();
    }
    cropper = new Cropper(imagePreview, {
        viewMode: 1,
        autoCropArea: 1,
        crop(event) {
            if (isUpdatingCrop) return;
            socket.emit('sync_action', { 
                room: roomId, 
                type: 'crop_change', 
                cropData: cropper.getData() 
            });
        }
    });
}

// Generic Sync Action Handler
socket.on('sync_action', (data) => {
    switch (data.type) {
        case 'draw_start':
            if (ctx) {
                ctx.beginPath();
                ctx.moveTo(data.x, data.y);
            }
            break;
        case 'draw_move':
            if (ctx) {
                ctx.lineWidth = 5;
                ctx.lineCap = "round";
                ctx.strokeStyle = data.color || '#000000';
                ctx.lineTo(data.x, data.y);
                ctx.stroke();
            }
            activeSource = 'canvas';
            break;
        case 'draw_end':
            if (ctx) {
                ctx.beginPath();
            }
            activeSource = 'canvas';
            break;
        case 'clear_canvas':
            if (ctx && canvas) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "white";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            }
            activeSource = 'none';
            break;
        case 'upload_image':
            const imagePreviewContainer = document.getElementById("imagePreviewContainer");
            const imagePreview = document.getElementById("imagePreview");
            if (imagePreview) {
                imagePreview.setAttribute("src", data.image);
                imagePreviewContainer.style.display = "block";
                
                if (cropper) cropper.destroy();
                setTimeout(initCropper, 100);
            }
            activeSource = 'upload';
            break;
        case 'crop_change':
            if (cropper && data.cropData) {
                isUpdatingCrop = true;
                cropper.setData(data.cropData);
                isUpdatingCrop = false;
            }
            break;
        case 'latex_edit':
            const latexBox = document.getElementById("latexBox");
            const mathOutput = document.getElementById("mathOutput");
            if (latexBox) latexBox.value = data.latex;
            if (mathOutput) mathOutput.innerHTML = data.latex.trim() ? "$$ " + data.latex + " $$" : "";
            if (window.MathJax) {
                MathJax.typesetPromise();
            }
            break;
        case 'prediction_start':
            toggleButtons(true, true);
            break;
        case 'prediction_end':
            toggleButtons(false, true);
            break;
    }
});

document.addEventListener('DOMContentLoaded', () => {
    const connectMobileBtn = document.getElementById('connectMobileBtn');
    if (connectMobileBtn) {
        connectMobileBtn.addEventListener('click', () => {
            const qrContainer = document.getElementById("qrcode");
            qrContainer.innerHTML = "";
            const url = window.location.origin + "/?room=" + roomId;
            new QRCode(qrContainer, {
                text: url,
                width: 150,
                height: 150
            });
            connectMobileBtn.style.display = "none";
        });
    }
});

// --- Canvas Drawing Logic ---
const canvas = document.getElementById('canvas');
const ctx = canvas ? canvas.getContext('2d') : null;
const rootStyles = window.getComputedStyle(document.documentElement);

// Resize canvas based on windows size changes
function resizeCanvas() {
    if (!canvas || !ctx) return;
    canvas.width = Math.min(window.innerWidth * 0.9, 800);
    canvas.height = Math.min(window.innerHeight * 0.5, 300);
    // Fill white background so transparent parts don't turn black
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

let painting = false;

if (canvas && ctx) {
    canvas.addEventListener("pointerdown", startDraw);
    canvas.addEventListener("pointerup", endDraw);
    canvas.addEventListener("pointerleave", endDraw);
    canvas.addEventListener("pointermove", draw);
}

function startDraw(e) {
    if (!ctx) return;
    painting = true;
    ctx.beginPath();
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    ctx.moveTo(x, y);
    socket.emit('sync_action', { room: roomId, type: 'draw_start', x: x, y: y });
    activeSource = 'canvas';
}

function endDraw() {
    if (!painting) return;
    painting = false;
    if (ctx) {
        ctx.beginPath(); // Reset path to prevent lines connecting when drawing again
        socket.emit('sync_action', { room: roomId, type: 'draw_end' });
    }
}

function draw(e) {
    if (!painting || !ctx) return;
    e.preventDefault();

    const color = rootStyles.getPropertyValue('--stroke_color').trim() || '#000000';
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.strokeStyle = color;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    ctx.lineTo(x, y);
    ctx.stroke();
    
    socket.emit('sync_action', { room: roomId, type: 'draw_move', x: x, y: y, color: color });
}

// Expose globally for the onclick handler in HTML
window.clearAll = function() {
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    activeSource = 'none';
    socket.emit('sync_action', { room: roomId, type: 'clear_canvas' });
};

function copyLatex() {
    const text = document.getElementById("latexBox");
    text.select();
    document.execCommand("copy");
    alert("Copied!");
}


const textarea = document.getElementById("latexBox");
const output = document.getElementById("mathOutput");

if (textarea) {
    textarea.addEventListener("input", function () {
        const latex = textarea.value;
        output.innerHTML = latex.trim() ? "$$ " + latex + " $$" : "";

        if (window.MathJax) {
            MathJax.typesetPromise();
        }
        
        socket.emit('sync_action', { room: roomId, type: 'latex_edit', latex: latex });
    });
}

const imageInput = document.getElementById("imageInput");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");
const imagePreview = document.getElementById("imagePreview");

if (imageInput) {
    imageInput.addEventListener("change", function () {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.addEventListener("load", function () {
                imagePreview.setAttribute("src", this.result);
                imagePreviewContainer.style.display = "block";
                activeSource = 'upload';
                socket.emit('sync_action', { room: roomId, type: 'upload_image', image: this.result });
                
                if (cropper) cropper.destroy();
                setTimeout(initCropper, 100);
            });
            reader.readAsDataURL(file);
        } else {
            imagePreviewContainer.style.display = "none";
            imagePreview.setAttribute("src", "");
            activeSource = 'none';
            if (cropper) {
                cropper.destroy();
                cropper = null;
            }
        }
    });
}

function toggleButtons(isPredicting, isRemote = false) {
    const predictBtn = document.getElementById('globalPredictBtn');
    const cancelBtn = document.getElementById('cancelPredictBtn');
    
    if (predictBtn) {
        predictBtn.disabled = isPredicting;
        predictBtn.textContent = isPredicting ? "Predicting..." : "Predict";
    }
    
    if (cancelBtn) {
        // Only show cancel button to the one who initiated the prediction
        if (isPredicting && !isRemote) {
            cancelBtn.style.display = "inline-block";
        } else {
            cancelBtn.style.display = "none";
        }
    }
}

function sendPredictionRequest(formData) {
    toggleButtons(true, false);
    socket.emit('sync_action', { room: roomId, type: 'prediction_start' });
    
    abortController = new AbortController();

    fetch("/predict", {
        method: "POST",
        body: formData,
        signal: abortController.signal
    })
    .then(response => response.json())
    .then(data => {
        toggleButtons(false, false);
        socket.emit('sync_action', { room: roomId, type: 'prediction_end' });
        
        if (data.latex !== undefined) {
            const latexBox = document.getElementById("latexBox");
            const mathOutput = document.getElementById("mathOutput");
            
            if (latexBox) latexBox.value = data.latex;
            if (mathOutput) mathOutput.innerHTML = data.latex.trim() ? "$$ " + data.latex + " $$" : "";
            
            if (window.MathJax) {
                MathJax.typesetPromise();
            }
            socket.emit('sync_prediction', { room: roomId, latex: data.latex });
        } else if (data.error) {
            alert("Error: " + data.error);
        }
    })
    .catch(err => {
        if (err.name === 'AbortError') {
            console.log('Prediction aborted.');
        } else {
            console.error("Error during fetch:", err);
            alert("An error occurred during prediction.");
        }
        toggleButtons(false, false);
        socket.emit('sync_action', { room: roomId, type: 'prediction_end' });
    });
}

// --- Unified Predict Logic ---
const globalPredictBtn = document.getElementById('globalPredictBtn');
if (globalPredictBtn) {
    globalPredictBtn.addEventListener('click', function() {
        if (activeSource === 'upload') {
            if (cropper) {
                // Get cropped region and send that
                cropper.getCroppedCanvas().toBlob(function(blob) {
                    const formData = new FormData();
                    formData.append('image', blob, 'cropped_image.png');
                    sendPredictionRequest(formData);
                }, 'image/png');
            } else {
                alert("No uploaded image or cropper found.");
            }
        } else if (activeSource === 'canvas') {
            if (!canvas) return;
            canvas.toBlob(function(blob) {
                const formData = new FormData();
                formData.append('image', blob, 'canvas_image.png');
                sendPredictionRequest(formData);
            }, 'image/png');
        } else {
            alert("Please upload an image or draw something first.");
        }
    });
}

const cancelPredictBtn = document.getElementById('cancelPredictBtn');
if (cancelPredictBtn) {
    cancelPredictBtn.addEventListener('click', function() {
        if (abortController) {
            abortController.abort();
            abortController = null;
        }
    });
}
