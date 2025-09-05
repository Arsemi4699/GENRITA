document.addEventListener('DOMContentLoaded', () => {

    // --- CLASS DEFINITIONS AND COLOR PALETTES ---
    const SENSE_CLASSES = {
        0: "Normal and neutral", 1: "Love and romantic", 2: "War and combat",
        3: "Fantasy and mythology", 4: "Honor and respect", 5: "Drama and tragedy",
        6: "City and Crowd", 7: "Mountain and the heights", 8: "Desert and dunes",
        9: "Sea and tides", 10: "Forest and tress",
    };

    const AGE_CLASSES = {
        0: "ancient and old age",
        1: "neutral and not special age (non-ancient, non technology)",
        2: "technology modern age",
    };

    const SENSE_COLORS_HSL = [
        [200, 30, 85], [340, 82, 85], [0, 79, 80], [270, 71, 85], [50, 80, 80],
        [240, 60, 85], [30, 60, 85], [180, 25, 80], [40, 50, 85], [210, 60, 80],
        [120, 40, 85],
    ];

    const AGE_COLORS_HSL = [
        [35, 30, 85], [210, 15, 90], [190, 70, 85],
    ];

    // --- DOM ELEMENT REFERENCES ---
    const titleElement = document.getElementById('file-title');
    const paragraphsContainer = document.getElementById('paragraphs-container');
    const modeToggle = document.getElementById('mode-toggle');
    const legendTitle = document.getElementById('legend-title');
    const legendList = document.getElementById('legend-list');
    const mainContent = document.getElementById('main-content');

    let currentParagraphData = null;

    // --- RENDER FUNCTIONS ---

    const renderRequestInfo = (data) => {
        const container = document.getElementById('request-info');
        if (!container) return;

        let paramsHTML = '<dl>';
        let abstractsHTML = '';
        let hasAbstracts = false;

        for (const [key, value] of Object.entries(data)) {
            if (key === 'target_abstracts' && typeof value === 'object' && Object.keys(value).length > 0) {
                hasAbstracts = true;
                for (const [abstractKey, abstractValue] of Object.entries(value)) {
                    abstractsHTML += `
                        <h4>${abstractKey.replace(/_/g, ' ')}</h4>
                        <p>${abstractValue}</p>
                    `;
                }
            } else {
                const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                paramsHTML += `<dt>${formattedKey}</dt><dd>${value}</dd>`;
            }
        }
        paramsHTML += '</dl>';

        if (hasAbstracts) {
            abstractsHTML = '<h3>Target Abstracts</h3>' + abstractsHTML;
        }

        container.innerHTML = `<h2>Request Details</h2>${paramsHTML}${abstractsHTML}`;
    };

    const renderParagraphs = (mode) => {
        if (!currentParagraphData) return;
        paragraphsContainer.innerHTML = '';
        legendList.innerHTML = '';

        const isAgeMode = mode === 'age';
        const classes = isAgeMode ? AGE_CLASSES : SENSE_CLASSES;
        const colors = isAgeMode ? AGE_COLORS_HSL : SENSE_COLORS_HSL;
        legendTitle.textContent = isAgeMode ? 'Age Classes' : 'Sense Classes';

        // Render Legend
        for (const classId in classes) {
            const className = classes[classId];
            const [h, s, l] = colors[classId];
            const li = document.createElement('li');
            const swatch = document.createElement('span');
            swatch.className = 'legend-color-swatch';
            swatch.style.backgroundColor = `hsl(${h}, ${s}%, ${l-15}%)`;
            li.appendChild(swatch);
            li.appendChild(document.createTextNode(className));
            legendList.appendChild(li);
        }

        // Render Paragraphs
        currentParagraphData.paragraphs.forEach(p => {
            const paragraphDiv = document.createElement('div');
            paragraphDiv.className = 'paragraph';
            const prediction = isAgeMode ? p.age_prediction : p.sense_prediction;
            const { class_id, confidence } = prediction;
            const [h, s, base_l] = colors[class_id];
            const lightness = 95 - (confidence * 25);
            paragraphDiv.style.backgroundColor = `hsl(${h}, ${s}%, ${lightness}%)`;
            paragraphDiv.innerHTML = generateParagraphHTML(p.text, p.entities);
            paragraphsContainer.appendChild(paragraphDiv);
        });
    };

    const generateParagraphHTML = (text, entities) => {
        if (!entities || entities.length === 0) return text;
        const sortedEntities = [...entities].sort((a, b) => a.start_pos - b.start_pos);
        let lastIndex = 0;
        let htmlContent = '';
        sortedEntities.forEach(entity => {
            htmlContent += text.slice(lastIndex, entity.start_pos);
            const entityText = text.slice(entity.start_pos, entity.end_pos);
            htmlContent += `<span class="entity">${entityText}<span class="tooltip">${entity.type}</span></span>`;
            lastIndex = entity.end_pos;
        });
        htmlContent += text.slice(lastIndex);
        return htmlContent;
    };

    // --- EVENT LISTENER ---
    modeToggle.addEventListener('change', (event) => {
        const mode = event.target.checked ? 'age' : 'sense';
        renderParagraphs(mode);
    });

    // --- INITIAL DATA FETCH AND RENDER ---
    Promise.all([
        fetch('data.json').then(res => res.json()),
        fetch('data_request.json').then(res => res.json())
    ])
    .then(([paragraphData, requestData]) => {
        currentParagraphData = paragraphData;
        titleElement.textContent = paragraphData.title || "Untitled Document";

        // Render both sections
        renderRequestInfo(requestData);
        renderParagraphs('sense'); // Initial render in 'sense' mode
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        mainContent.innerHTML = '<p>Failed to load data. Please check the console and ensure both data.json and data_request.json exist.</p>';
    });
});