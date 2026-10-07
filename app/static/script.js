const imageInput = document.getElementById(
    "imageInput"
);

const chooseButton = document.getElementById(
    "chooseButton"
);

const dropZone = document.getElementById(
    "dropZone"
);

const selectedFile = document.getElementById(
    "selectedFile"
);

const sampleGrid = document.getElementById(
    "sampleGrid"
);

const confidenceSlider =
    document.getElementById(
        "confidenceSlider"
    );

const confidenceValue =
    document.getElementById(
        "confidenceValue"
    );

const resultSection =
    document.getElementById(
        "resultSection"
    );

const resultImage =
    document.getElementById(
        "resultImage"
    );

const resultStatus =
    document.getElementById(
        "resultStatus"
    );

const detectionCount =
    document.getElementById(
        "detectionCount"
    );

const resultThreshold =
    document.getElementById(
        "resultThreshold"
    );

const resultFilename =
    document.getElementById(
        "resultFilename"
    );

const resultDimensions =
    document.getElementById(
        "resultDimensions"
    );

const detectionList =
    document.getElementById(
        "detectionList"
    );

const modelStatus =
    document.getElementById(
        "modelStatus"
    );


function currentConfidence() {

    return (
        Number(
            confidenceSlider.value
        ) / 100
    );
}


confidenceSlider.addEventListener(
    "input",
    () => {

        confidenceValue.textContent =
            confidenceSlider.value;

    }
);


chooseButton.addEventListener(
    "click",
    () => {

        imageInput.click();

    }
);


imageInput.addEventListener(
    "change",
    () => {

        if (
            !imageInput.files.length
        ) {
            return;
        }

        const file =
            imageInput.files[0];

        selectedFile.textContent =
            file.name;

        runUploadPrediction(
            file
        );
    }
);


[
    "dragenter",
    "dragover"
].forEach(
    (eventName) => {

        dropZone.addEventListener(
            eventName,
            (event) => {

                event.preventDefault();

                dropZone.classList.add(
                    "dragover"
                );
            }
        );
    }
);


[
    "dragleave",
    "drop"
].forEach(
    (eventName) => {

        dropZone.addEventListener(
            eventName,
            (event) => {

                event.preventDefault();

                dropZone.classList.remove(
                    "dragover"
                );
            }
        );
    }
);


dropZone.addEventListener(
    "drop",
    (event) => {

        const file =
            event.dataTransfer.files[0];

        if (!file) {
            return;
        }

        selectedFile.textContent =
            file.name;

        runUploadPrediction(
            file
        );
    }
);


async function loadSamples() {

    try {

        const response =
            await fetch("/samples");

        const samples =
            await response.json();

        sampleGrid.innerHTML = "";

        if (
            !samples.length
        ) {

            sampleGrid.innerHTML =
                `
                <div class="sample-loading">
                    No sample images found.
                </div>
                `;

            return;
        }

        samples.forEach(
            (sample) => {

                const button =
                    document.createElement(
                        "button"
                    );

                button.type = "button";

                button.className =
                    "sample-button";

                button.innerHTML = `
                    <img
                        src="${sample.url}"
                        alt="${sample.filename}"
                    >

                    <span>
                        ${sample.filename}
                    </span>
                `;

                button.addEventListener(
                    "click",
                    () => {

                        runSamplePrediction(
                            sample.filename
                        );

                    }
                );

                sampleGrid.appendChild(
                    button
                );
            }
        );

    } catch (error) {

        sampleGrid.innerHTML =
            `
            <div class="sample-loading">
                Could not load samples.
            </div>
            `;
    }
}


async function runUploadPrediction(
    file
) {

    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    resultSection.classList.remove(
        "hidden"
    );

    resultStatus.textContent =
        "Running detection...";

    try {

        const response =
            await fetch(
                `/predict?confidence=${currentConfidence()}`,
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Prediction failed."
            );
        }

        renderResult(
            data
        );

    } catch (error) {

        resultStatus.textContent =
            error.message;
    }
}


async function runSamplePrediction(
    filename
) {

    resultSection.classList.remove(
        "hidden"
    );

    resultStatus.textContent =
        `Running detection on ${filename}...`;

    try {

        const response =
            await fetch(
                `/predict-sample/${encodeURIComponent(filename)}?confidence=${currentConfidence()}`,
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Sample prediction failed."
            );
        }

        renderResult(
            data
        );

    } catch (error) {

        resultStatus.textContent =
            error.message;
    }
}


function renderResult(
    data
) {

    resultImage.src =
        data.annotated_image;

    resultStatus.textContent =
        data.status;

    detectionCount.textContent =
        data.detection_count;

    resultThreshold.textContent =
        `${confidenceSlider.value}%`;

    resultFilename.textContent =
        data.filename;

    resultDimensions.textContent =
        `${data.image_width} × ${data.image_height}`;

    detectionList.innerHTML = "";

    if (
        data.detections.length === 0
    ) {

        detectionList.innerHTML =
            `
            <div class="empty-state">
                No FOD objects crossed
                the selected confidence threshold.
            </div>
            `;

        return;
    }

    data.detections.forEach(
        (detection, index) => {

            const item =
                document.createElement(
                    "div"
                );

            item.className =
                "detection-item";

            item.innerHTML = `
                <div>

                    <div class="detection-name">
                        ${index + 1}.
                        ${detection.class_name}
                    </div>

                    <div class="detection-box">
                        Box:
                        (${detection.xmin.toFixed(1)},
                        ${detection.ymin.toFixed(1)})
                        →
                        (${detection.xmax.toFixed(1)},
                        ${detection.ymax.toFixed(1)})
                    </div>

                </div>

                <div class="detection-confidence">
                    ${(detection.confidence * 100).toFixed(1)}%
                </div>
            `;

            detectionList.appendChild(
                item
            );
        }
    );
}


async function checkModelHealth() {

    try {

        const response =
            await fetch(
                "/health"
            );

        const data =
            await response.json();

        if (
            response.ok &&
            data.model_loaded
        ) {

            modelStatus.textContent =
                "ONLINE";

        } else {

            modelStatus.textContent =
                "CHECK FAILED";
        }

    } catch (error) {

        modelStatus.textContent =
            "OFFLINE";
    }
}


loadSamples();

checkModelHealth();