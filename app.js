// app.js - Titanic Binary Classifier with TensorFlow.js
// ======================================================

// ===========================================================================
// GLOBAL STATE
// ============================================================================
let state = {
    trainData: null,
    testData: null,
    trainTensors: null,
    testTensors: null,
    model: null,
    featureNames: [],
    featureImportance: null,
    predictions: null,
    validationProbs: null,
    validationLabels: null,
    rocData: null,
    auc: 0,
    isTrained: false
};
let trainIndices = [];
let valIndices = [];


// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Robust CSV parsing with quote handling
 * @param {string} text - Raw CSV content
 * @returns {Array<Array<string>>} Parsed rows
 */
function parseCSV(text) {
    const rows = [];
    let currentRow = [];
    let currentField = '';
    let insideQuotes = false;
    
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        const nextChar = text[i + 1] || '';
        
        if (char === '"') {
            if (insideQuotes && nextChar === '"') {
                currentField += '"';
                i++; // Skip next quote
            } else {
                insideQuotes = !insideQuotes;
            }
        } else if (char === ',' && !insideQuotes) {
            currentRow.push(currentField.trim());
            currentField = '';
        } else if (char === '\n' && !insideQuotes) {
            currentRow.push(currentField.trim());
            rows.push(currentRow);
            currentRow = [];
            currentField = '';
            
            // Handle Windows line endings
            if (nextChar === '\r') i++;
        } else if (char === '\r' && !insideQuotes) {
            // Ignore carriage returns unless inside quotes
            continue;
        } else {
            currentField += char;
        }
    }
    
    // Add last row
    if (currentField.trim() !== '' || currentRow.length > 0) {
        currentRow.push(currentField.trim());
        rows.push(currentRow);
    }
    
    return rows;
}

/**
 * Convert CSV rows to object array
 * @param {Array<Array<string>>} rows - Parsed CSV rows
 * @returns {Array<Object>} Array of objects
 */
function rowsToObjects(rows) {
    if (rows.length < 2) return [];
    
    const headers = rows[0];
    const data = [];
    
    for (let i = 1; i < rows.length; i++) {
        if (rows[i].length !== headers.length) {
            console.warn(`Row ${i} has ${rows[i].length} columns, expected ${headers.length}`);
            continue;
        }
        
        const obj = {};
        for (let j = 0; j < headers.length; j++) {
            let value = rows[i][j];
            // Try to convert numeric values
            if (!isNaN(value) && value.trim() !== '') {
                value = parseFloat(value);
            } else if (value === '') {
                value = null;
            }
            obj[headers[j]] = value;
        }
        data.push(obj);
    }
    
    return data;
}

/**
 * Update status message
 * @param {string} id - Element ID
 * @param {string} message - Status message
 * @param {string} type - 'normal', 'success', or 'error'
 */
function updateStatus(id, message, type = 'normal') {
    const element = document.getElementById(id);
    if (!element) return;
    
    element.textContent = message;
    element.className = 'status';
    if (type === 'success') element.classList.add('success');
    if (type === 'error') element.classList.add('error');
}

/**
 * Create a simple table from data
 * @param {Array<Object>} data - Data to display
 * @param {number} maxRows - Maximum rows to show
 * @returns {string} HTML table
 */
function createTable(data, maxRows = 10) {
    if (!data || data.length === 0) return '<p>No data to display.</p>';
    
    const headers = Object.keys(data[0]);
    const rows = data.slice(0, maxRows);
    
    let html = '<table><thead><tr>';
    headers.forEach(h => html += `<th>${h}</th>`);
    html += '</tr></thead><tbody>';
    
    rows.forEach(row => {
        html += '<tr>';
        headers.forEach(h => {
            const val = row[h];
            html += `<td>${val !== null && val !== undefined ? val : '<em>null</em>'}</td>`;
        });
        html += '</tr>';
    });
    
    if (data.length > maxRows) {
        html += `<tr><td colspan="${headers.length}" style="text-align: center; font-style: italic;">... and ${data.length - maxRows} more rows</td></tr>`;
    }
    
    html += '</tbody></table>';
    return html;
}

/**
 * Calculate missing value percentages
 * @param {Array<Object>} data - Dataset
 * @returns {Object} Missing percentages per column
 */
function calculateMissing(data) {
    if (!data || data.length === 0) return {};
    
    const counts = {};
    const total = data.length;
    
    Object.keys(data[0]).forEach(col => {
        const missing = data.filter(row => row[col] === null || row[col] === undefined || row[col] === '').length;
        counts[col] = ((missing / total) * 100).toFixed(1);
    });
    
    return counts;
}

// ============================================================================
// DATA LOADING
// ============================================================================

/**
 * Load CSV file from input or URL
 */
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile) {
        alert('Please select training data file (train.csv)');
        return;
    }
    
    updateStatus('load-status', 'Loading data...');
    
    try {
        // Load training data
        const trainText = await trainFile.text();
        const trainRows = parseCSV(trainText);
        state.trainData = rowsToObjects(trainRows);
        
        // Load test data if provided
        if (testFile) {
            const testText = await testFile.text();
            const testRows = parseCSV(testText);
            state.testData = rowsToObjects(testRows);
        }
        
        updateStatus('load-status', `Loaded ${state.trainData.length} training samples${state.testData ? ` and ${state.testData.length} test samples` : ''}`, 'success');
        
        // Run inspection
        inspectData();
        
        // Enable preprocessing button
        document.getElementById('preprocess-btn').disabled = false;
        
    } catch (error) {
        updateStatus('load-status', `Error loading data: ${error.message}`, 'error');
        console.error(error);
    }
}

/**
 * Inspect loaded data
 */
function inspectData() {
    if (!state.trainData) return;
    
    // Display shape
    const cols = Object.keys(state.trainData[0]).length;
    document.getElementById('shape-info').innerHTML = `
        <p><strong>Training:</strong> ${state.trainData.length} rows × ${cols} columns</p>
        ${state.testData ? `<p><strong>Test:</strong> ${state.testData.length} rows × ${Object.keys(state.testData[0]).length} columns</p>` : ''}
    `;
    
    // Display missing values
    const missing = calculateMissing(state.trainData);
    let missingHTML = '<ul>';
    Object.entries(missing).forEach(([col, pct]) => {
        missingHTML += `<li><strong>${col}:</strong> ${pct}% missing</li>`;
    });
    missingHTML += '</ul>';
    document.getElementById('missing-info').innerHTML = missingHTML;
    
    // Display preview
    document.getElementById('data-preview').innerHTML = createTable(state.trainData, 8);
    
    // Create visualizations
    createVisualizations();
}

/**
 * Create tfjs-vis visualizations
 */
function createVisualizations() {
    if (!state.trainData) return;
    
    const visContainer = document.getElementById('vis-container');
    const surface = tfvis.visor().surface({ name: 'Data Distributions', tab: 'Training Data' });
    
    // Survival by Sex
    const survivalBySex = {};
    state.trainData.forEach(row => {
        if (row.Sex && row.Survived !== null && row.Survived !== undefined) {
            const key = `${row.Sex}-${row.Survived}`;
            if (!survivalBySex[key]) survivalBySex[key] = 0;
            survivalBySex[key]++;
        }
    });
    
    const sexData = [
        { index: '0', value: survivalBySex['male-0'] || 0, series: 'Male - Died' },
        { index: '1', value: survivalBySex['male-1'] || 0, series: 'Male - Survived' },
        { index: '2', value: survivalBySex['female-0'] || 0, series: 'Female - Died' },
        { index: '3', value: survivalBySex['female-1'] || 0, series: 'Female - Survived' }
    ].filter(d => d.value > 0);
    
    // Survival by Pclass
    const survivalByClass = {};
    state.trainData.forEach(row => {
        if (row.Pclass && row.Survived !== null && row.Survived !== undefined) {
            const key = `Class ${row.Pclass}-${row.Survived}`;
            if (!survivalByClass[key]) survivalByClass[key] = 0;
            survivalByClass[key]++;
        }
    });
    
    const classData = [];
    [1, 2, 3].forEach(pclass => {
        ['Died', 'Survived'].forEach(status => {
            const key = `Class ${pclass}-${status === 'Survived' ? 1 : 0}`;
            if (survivalByClass[key]) {
                classData.push({
                    index: `Class ${pclass}`,
                    value: survivalByClass[key],
                    series: status
                });
            }
        });
    });
    
    // Render charts
    tfvis.render.barchart(surface, sexData, {
        xLabel: 'Category',
        yLabel: 'Count',
        width: 400,
        height: 300
    });
    
    setTimeout(() => {
        const surface2 = tfvis.visor().surface({ name: 'Survival by Class', tab: 'Training Data' });
        tfvis.render.barchart(surface2, classData, {
            xLabel: 'Passenger Class',
            yLabel: 'Count',
            width: 400,
            height: 300
        });
    }, 100);
}

// ============================================================================
// PREPROCESSING
// ============================================================================

/**
 * Preprocess data with feature engineering
 */
function preprocessData() {
    if (!state.trainData) {
        alert('Load data first');
        return;
    }
    
    updateStatus('preprocess-status', 'Preprocessing data...');
    
    // ============================================================================
    // SCHEMA DEFINITION - SWAP THESE FOR OTHER DATASETS
    // ============================================================================
    const TARGET_COLUMN = 'Survived';            // Binary target variable
    const NUMERIC_FEATURES = ['Age', 'Fare'];   // Features to standardize
    const CATEGORICAL_FEATURES = ['Sex', 'Pclass', 'Embarked']; // Features to one-hot encode
    const ID_COLUMN = 'PassengerId';            // Identifier (excluded from training)
    // ============================================================================
    
    // Combine train and test for consistent preprocessing
    const allData = [...state.trainData];
    const testData = state.testData ? [...state.testData] : [];
    const hasTestData = testData.length > 0;
    
    if (hasTestData) {
        allData.push(...testData);
    }
    
    // Calculate imputation values from training data only
    const trainOnly = state.trainData.filter(d => d[TARGET_COLUMN] !== null);
    
    // Median imputation for Age
    const ages = trainOnly.map(d => d.Age).filter(a => a !== null && !isNaN(a));
    const ageMedian = ages.length > 0 ? ages.sort((a, b) => a - b)[Math.floor(ages.length / 2)] : 0;
    
    // Mode imputation for Embarked
    const embarkedCounts = {};
    trainOnly.forEach(d => {
        if (d.Embarked && d.Embarked !== null) {
            embarkedCounts[d.Embarked] = (embarkedCounts[d.Embarked] || 0) + 1;
        }
    });
    const embarkedMode = Object.keys(embarkedCounts).length > 0 
        ? Object.keys(embarkedCounts).reduce((a, b) => embarkedCounts[a] > embarkedCounts[b] ? a : b)
        : 'S';
    
    // Calculate standardization parameters from training data
    const fareValues = trainOnly.map(d => d.Fare).filter(f => f !== null && !isNaN(f));
    const fareMean = fareValues.length > 0 ? fareValues.reduce((a, b) => a + b, 0) / fareValues.length : 0;
    const fareStd = fareValues.length > 0 
        ? Math.sqrt(fareValues.map(f => Math.pow(f - fareMean, 2)).reduce((a, b) => a + b, 0) / fareValues.length)
        : 1;
    
    // Apply preprocessing to all data
    const processed = allData.map(row => {
        const processedRow = { ...row };
        
        // Impute Age
        if (processedRow.Age === null || isNaN(processedRow.Age)) {
            processedRow.Age = ageMedian;
        }
        
        // Impute Embarked
        if (!processedRow.Embarked || processedRow.Embarked === null) {
            processedRow.Embarked = embarkedMode;
        }
        
        // Standardize Age and Fare
        processedRow.Age_std = (processedRow.Age - ageMedian) / (ageMedian || 1);
        if (processedRow.Fare !== null && !isNaN(processedRow.Fare)) {
            processedRow.Fare_std = (processedRow.Fare - fareMean) / (fareStd || 1);
        } else {
            processedRow.Fare_std = 0;
        }
        
        // Feature engineering (optional)
        const addFamilySize = document.getElementById('feat-family').checked;
        const addIsAlone = document.getElementById('feat-alone').checked;
        
        if (addFamilySize) {
            processedRow.FamilySize = (processedRow.SibSp || 0) + (processedRow.Parch || 0) + 1;
        }
        
        if (addIsAlone) {
            const familySize = (processedRow.SibSp || 0) + (processedRow.Parch || 0) + 1;
            processedRow.IsAlone = familySize === 1 ? 1 : 0;
        }
        
        return processedRow;
    });
    
    // Split back into train and test
    const processedTrain = processed.slice(0, state.trainData.length);
    const processedTest = hasTestData ? processed.slice(state.trainData.length) : [];
    
    // Prepare feature list
    let featureList = [];
    
    // Add standardized numeric features
    featureList.push('Age_std', 'Fare_std');
    
    // Add original numeric features (excluding those standardized)
    NUMERIC_FEATURES.forEach(feat => {
        if (!['Age', 'Fare'].includes(feat)) {
            featureList.push(feat);
        }
    });
    
    // Add engineered features
    if (document.getElementById('feat-family').checked) featureList.push('FamilySize');
    if (document.getElementById('feat-alone').checked) featureList.push('IsAlone');
    
    // Add SibSp and Parch if not already included
    if (!featureList.includes('SibSp')) featureList.push('SibSp');
    if (!featureList.includes('Parch')) featureList.push('Parch');
    
    // One-hot encode categorical features
    const categoricalMaps = {};
    CATEGORICAL_FEATURES.forEach(feat => {
        const uniqueVals = [...new Set(processedTrain.map(r => r[feat]).filter(v => v !== null))];
        categoricalMaps[feat] = uniqueVals;
        
        uniqueVals.forEach(val => {
            featureList.push(`${feat}_${val}`);
        });
    });
    
    // Convert to tensors
    const { features: trainFeatures, labels: trainLabels } = dataToTensors(processedTrain, TARGET_COLUMN, featureList, categoricalMaps, ID_COLUMN);
    const testFeatures = hasTestData 
        ? dataToTensors(processedTest, null, featureList, categoricalMaps, ID_COLUMN).features
        : null;
    const testIds = hasTestData ? processedTest.map(r => r[ID_COLUMN]) : null;
    
    // Store tensors
    state.trainTensors = {
        features: trainFeatures,
        labels: trainLabels,
        ids: processedTrain.map(r => r[ID_COLUMN])
    };
    
    if (hasTestData) {
        state.testTensors = {
            features: testFeatures,
            ids: testIds
        };
    }
    
    // Update UI
    state.featureNames = featureList;
    document.getElementById('input-dim').textContent = featureList.length;
    
    updateStatus('preprocess-status', `Preprocessed ${featureList.length} features`, 'success');
    
    document.getElementById('feature-info').innerHTML = `
        <p><strong>Features:</strong> ${featureList.join(', ')}</p>
        <p><strong>Train tensors:</strong> ${trainFeatures.shape}</p>
        <p><strong>Train labels:</strong> ${trainLabels.shape}</p>
    `;
    
    // Enable training button
    document.getElementById('train-btn').disabled = false;
}

/**
 * Convert processed data to tensors
 */
function dataToTensors(data, targetColumn, featureList, categoricalMaps, idColumn) {
    const features = [];
    const labels = [];
    
    data.forEach(row => {
        // Extract features
        const featureVec = [];
        
        featureList.forEach(feat => {
            if (feat.includes('_') && categoricalMaps) {
                // One-hot encoded feature
                const [catName, catVal] = feat.split('_');
                featureVec.push(row[catName] === catVal ? 1 : 0);
            } else if (row[feat] !== null && !isNaN(row[feat])) {
                // Numeric feature
                featureVec.push(row[feat]);
            } else {
                // Missing numeric feature
                featureVec.push(0);
            }
        });
        
        features.push(featureVec);
        
        // Extract label if available
        if (targetColumn && row[targetColumn] !== null && !isNaN(row[targetColumn])) {
            labels.push(row[targetColumn]);
        }
    });
    
    return {
        features: tf.tensor2d(features),
        labels: targetColumn ? tf.tensor2d(labels, [labels.length, 1]) : null
    };
}

// ============================================================================
// MODEL DEFINITION
// ============================================================================

/**
 * Create model with sigmoid feature gate
 */
function createModel() {
    if (!state.trainTensors) {
        alert('Preprocess data first');
        return;
    }
    
    const inputDim = state.featureNames.length;
    
    // ============================================================================
    // MODEL ARCHITECTURE - ADJUST FOR OTHER PROBLEMS
    // ============================================================================
    state.model = tf.sequential();
    
    // Sigmoid feature gate layer (learnable feature importance)
    state.model.add(tf.layers.dense({
        units: inputDim,
        activation: 'sigmoid',
        useBias: false,
        kernelInitializer: 'ones',
        inputShape: [inputDim],
        name: 'feature_gate'
    }));
    
    // Hidden layer
    state.model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        kernelInitializer: 'heNormal',
        name: 'hidden'
    }));
    
    // Output layer
    state.model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        name: 'output'
    }));
    
    // Compile model
    state.model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const summaryContainer = document.getElementById('model-summary');
    summaryContainer.innerHTML = '<h4>Model Summary</h4>';
    tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, state.model);
    
    return state.model;
}

// ============================================================================
// TRAINING
// ============================================================================

/**
 * Train the model
 */
async function trainModel() {
    if (!state.model) {
        createModel();
    }
    
    if (!state.trainTensors) {
        alert('Preprocess data first');
        return;
    }
    
    updateStatus('train-status', 'Training model...');
    
    const features = state.trainTensors.features;
    const labels = state.trainTensors.labels;
    
    // Split into training and validation (80/20 stratified)
    const indices = tf.util.createShuffledIndices(features.shape[0]);
    const splitIdx = Math.floor(features.shape[0] * 0.8);
    
    trainIndices = Array.from(indices.slice(0, splitIdx));
    valIndices = Array.from(indices.slice(splitIdx));

    const trainIdxTensor = tf.tensor1d(trainIndices, 'int32')
    const valIdxTensor = tf.tensor1d(valIndices, 'int32')

    let trainFeatures = tf.gather(features, trainIdxTensor)
    let trainLabels = tf.gather(labels, trainIdxTensor)
    let valFeatures = tf.gather(features, valIdxTensor)
    let valLabels = tf.gather(labels, valIdxTensor)

    trainLabels = trainLabels.reshape([-1,1])
    valLabels = valLabels.reshape([-1,1])
    
    // Store validation data for ROC calculation
    state.validationProbs = null;
    state.validationLabels = valLabels.arraySync().map(v => v[0]);
    
    // Training configuration
    const epochs = 50;
    const batchSize = 32;
    let bestValLoss = Infinity;
    let patience = 5;
    let patienceCounter = 0;
    
    // Callbacks for visualization and early stopping
    const callbacks = tfvis.show.fitCallbacks(
        { name: 'Training Metrics', tab: 'Training' },
        ['loss', 'val_loss', 'acc', 'val_acc'],
        {
            callbacks: ['onEpochEnd'],
            height: 300
        }
    );
    
    // Custom callback for early stopping
    const earlyStoppingCallback = {
        onEpochEnd: async (epoch, logs) => {
            // Early stopping logic
            if (logs.val_loss < bestValLoss) {
                bestValLoss = logs.val_loss;
                patienceCounter = 0;
            } else {
                patienceCounter++;
                if (patienceCounter >= patience) {
                    state.model.stopTraining = true;
                    console.log(`Early stopping at epoch ${epoch + 1}`);
                }
            }
            
            // Get validation predictions for ROC
            const valPreds = state.model.predict(valFeatures);
            state.validationProbs = await valPreds.data();
            valPreds.dispose();
        }
    };
    
    // Combine callbacks
    const allCallbacks = {
        onEpochEnd: (epoch, logs) => {
            if (callbacks && callbacks.onEpochEnd) {
                callbacks.onEpochEnd(epoch, logs);
            }
            earlyStoppingCallback.onEpochEnd(epoch, logs);
        }
    };
    
    try {
        // Train the model
        const history = await state.model.fit(trainFeatures, trainLabels, {
            epochs,
            batchSize,
            validationData: [valFeatures, valLabels],
            callbacks: allCallbacks,
            verbose: 0
        });
        
        // Clean up tensors
        trainFeatures.dispose();
        trainLabels.dispose();
        valFeatures.dispose();
        valLabels.dispose();
        
        // Extract feature importance from gate layer (diagonal of gate matrix)
        const gateLayer = state.model.getLayer('feature_gate');
        const gateWeights = gateLayer.getWeights()[0];

        const gateKernel = gateLayer.getWeights()[0];

        // APPLY SIGMOID TO KERNEL
        const gateSigmoid = tf.sigmoid(gateKernel);
        const gateArr = await gateSigmoid.array();

        // diagonal = feature importance
        state.featureImportance = gateArr.map((row, i) => Number(row[i]));
        
        state.isTrained = true;
        updateStatus('train-status', `Training completed! Final val_accuracy: ${history.history.val_acc[history.history.val_acc.length - 1].toFixed(4)}`, 'success');
        
        // Enable evaluation and prediction
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('predict-btn').disabled = !state.testTensors;
        
        // Run evaluation
        evaluateModel();
        renderFeatureImportance();
        
    } catch (error) {
        updateStatus('train-status', `Training error: ${error.message}`, 'error');
        console.error(error);
    }
}

// ============================================================================
// EVALUATION
// ============================================================================

/**
 * Evaluate model and update metrics
 */
async function evaluateModel() {
    if (!state.model || !state.validationProbs || !state.validationLabels) return;
    
    // Calculate ROC curve with 100 thresholds
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocPoints = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fp = 0, tn = 0, fn = 0;
        
        state.validationProbs.forEach((prob, i) => {
            const pred = prob >= threshold ? 1 : 0;
            const actual = state.validationLabels[i];
            
            if (pred === 1 && actual === 1) tp++;
            else if (pred === 1 && actual === 0) fp++;
            else if (pred === 0 && actual === 0) tn++;
            else if (pred === 0 && actual === 1) fn++;
        });
        
        const tpr = tp + fn > 0 ? tp / (tp + fn) : 0;
        const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
        
        rocPoints.push({ threshold, tpr, fpr });
    });
    
    // Sort by FPR for AUC calculation
    rocPoints.sort((a, b) => a.fpr - b.fpr);
    
    // Calculate AUC using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        const prev = rocPoints[i - 1];
        const curr = rocPoints[i];
        auc += (curr.fpr - prev.fpr) * (curr.tpr + prev.tpr) / 2;
    }
    
    // Store for later use
    state.rocData = rocPoints;
    state.auc = auc;
    
    // Plot ROC curve
    const rocVisData = rocPoints.map(p => ({ x: p.fpr, y: p.tpr }));
    const rocSurface = tfvis.visor().surface({ name: 'ROC Curve', tab: 'Evaluation' });
    
    tfvis.render.linechart(
        rocSurface,
        {
            values: [rocVisData],
            series: ['ROC']
        },
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            width: 400,
            height: 300
        }
    );
    
    // Update metrics display
    updateThresholdDisplay();
    
    // Display feature importance
    displayFeatureImportance();
    
    // Update AUC in UI
    const aucElement = document.getElementById('auc-display');
    if (aucElement) {
        aucElement.textContent = `AUC: ${auc.toFixed(3)}`;
    }
}

/**
 * Update metrics based on current threshold
 */
function updateThresholdDisplay() {
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    if (!state.validationProbs || !state.validationLabels) return;
    
    // Calculate confusion matrix
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    state.validationProbs.forEach((prob, i) => {
        const pred = prob >= threshold ? 1 : 0;
        const actual = state.validationLabels[i];
        
        if (pred === 1 && actual === 1) tp++;
        else if (pred === 1 && actual === 0) fp++;
        else if (pred === 0 && actual === 0) tn++;
        else if (pred === 0 && actual === 1) fn++;
    });
    
    // Calculate metrics
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    
    // Update metrics display
    const metricsHTML = `
        <div class="metric-card">
            <div class="metric-value">${accuracy.toFixed(3)}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${precision.toFixed(3)}</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${recall.toFixed(3)}</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${f1.toFixed(3)}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">${state.auc.toFixed(3)}</div>
            <div class="metric-label">AUC</div>
        </div>
    `;
    
    document.getElementById('metrics-display').innerHTML = metricsHTML;
    
    // Update evaluation status
    document.getElementById('eval-status').innerHTML = `
        <strong>Confusion Matrix (Threshold = ${threshold.toFixed(2)})</strong><br>
        <table style="width: auto; margin-top: 10px;">
            <tr><td></td><td><strong>Predicted 0</strong></td><td><strong>Predicted 1</strong></td></tr>
            <tr><td><strong>Actual 0</strong></td><td>${tn}</td><td>${fp}</td></tr>
            <tr><td><strong>Actual 1</strong></td><td>${fn}</td><td>${tp}</td></tr>
        </table>
    `;
}

/**
 * Display feature importance from sigmoid gate
 */
function displayFeatureImportance() {
    if (!state.featureImportance || !state.featureNames) return;
    
    // Create table
    let tableHTML = '<table><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>';
    
    state.featureImportance.forEach((importance, i) => {
        const featureName = state.featureNames[i] || `Feature ${i}`;
        tableHTML += `<tr><td>${featureName}</td><td>${importance.toFixed(4)}</td></tr>`;
    });
    
    tableHTML += '</tbody></table>';
    document.getElementById('importance-table').innerHTML = tableHTML;
    
    // Create bar chart
    const importanceData = state.featureImportance.map((imp, i) => ({
        index: state.featureNames[i] || `F${i}`,
        value: imp
    }));
    
    const importanceSurface = tfvis.visor().surface({ name: 'Feature Importance', tab: 'Evaluation' });
    tfvis.render.barchart(importanceSurface, importanceData, {
        xLabel: 'Feature',
        yLabel: 'Importance',
        width: 500,
        height: 300
    });
}

// ============================================================================
// PREDICTION
// ============================================================================

/**
 * Generate predictions on test set
 */
async function generatePredictions() {
    if (!state.model || !state.isTrained) {
        alert('Train model first');
        return;
    }
    
    if (!state.testTensors) {
        alert('No test data loaded');
        return;
    }
    
    updateStatus('predict-status', 'Generating predictions...');
    
    try {
        // Get predictions
        const probsTensor = state.model.predict(state.testTensors.features);
        const probabilities = await probsTensor.data();
        probsTensor.dispose();
        
        // Apply threshold
        const threshold = parseFloat(document.getElementById('threshold-slider').value);
        const predictions = probabilities.map(p => p >= threshold ? 1 : 0);
        
        // Store results
        state.predictions = {
            ids: state.testTensors.ids,
            probabilities: Array.from(probabilities),
            predictions: predictions
        };
        
        // Display preview
        const previewData = [];
        const maxPreview = Math.min(10, state.predictions.ids.length);
        
        for (let i = 0; i < maxPreview; i++) {
            previewData.push({
                PassengerId: state.predictions.ids[i],
                Probability: state.predictions.probabilities[i].toFixed(4),
                Survived: state.predictions.predictions[i]
            });
        }
        
        document.getElementById('prediction-preview').innerHTML = createTable(previewData);
        
        updateStatus('predict-status', `Generated ${predictions.length} predictions`, 'success');
        
        // Enable export buttons
        document.getElementById('export-submission').disabled = false;
        document.getElementById('export-probabilities').disabled = false;
        document.getElementById('export-model').disabled = false;
        
    } catch (error) {
        updateStatus('predict-status', `Prediction error: ${error.message}`, 'error');
        console.error(error);
    }
}

// ============================================================================
// EXPORT FUNCTIONS
// ============================================================================

/**
 * Download CSV file
 */
function downloadCSV(filename, content) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
function renderFeatureImportance() {
    const div = document.getElementById('importance-table');
 
    let html = '<table><tr><th>Feature</th><th>Importance</th></tr>';

    state.featureNames.forEach((f,i)=>{
    html += `<tr><td>${f}</td><td>${state.featureImportance[i].toFixed(4)}</td></tr>`;
    });

    html += '</table>';
    div.innerHTML = html;
}


/**
 * Export submission CSV
 */
function exportSubmission() {
    if (!state.predictions) {
        alert('Generate predictions first');
        return;
    }
    
    let csv = 'PassengerId,Survived\n';
    
    state.predictions.ids.forEach((id, i) => {
        csv += `${id},${state.predictions.predictions[i]}\n`;
    });
    
    downloadCSV('submission.csv', csv);
    updateStatus('export-status', 'Downloaded submission.csv', 'success');
}

/**
 * Export probabilities CSV
 */
function exportProbabilities() {
    if (!state.predictions) {
        alert('Generate predictions first');
        return;
    }
    
    let csv = 'PassengerId,Probability\n';
    
    state.predictions.ids.forEach((id, i) => {
        csv += `${id},${state.predictions.probabilities[i].toFixed(6)}\n`;
    });
    
    downloadCSV('probabilities.csv', csv);
    updateStatus('export-status', 'Downloaded probabilities.csv', 'success');
}

/**
 * Export trained model
 */
async function exportModel() {
    if (!state.model || !state.isTrained) {
        alert('Train model first');
        return;
    }
    
    try {
        await state.model.save('downloads://titanic-tfjs-model');
        updateStatus('export-status', 'Model download started', 'success');
    } catch (error) {
        updateStatus('export-status', `Export error: ${error.message}`, 'error');
    }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Load data button
    document.getElementById('load-btn').addEventListener('click', loadData);
    
    // Preprocess button
    document.getElementById('preprocess-btn').addEventListener('click', preprocessData);
    
    // Train button
    document.getElementById('train-btn').addEventListener('click', trainModel);
    
    // Threshold slider
    document.getElementById('threshold-slider').addEventListener('input', updateThresholdDisplay);
    
    // Predict button
    document.getElementById('predict-btn').addEventListener('click', generatePredictions);
    
    // Export buttons
    document.getElementById('export-submission').addEventListener('click', exportSubmission);
    document.getElementById('export-probabilities').addEventListener('click', exportProbabilities);
    document.getElementById('export-model').addEventListener('click', exportModel);
});

// ============================================================================
// CODE SUMMARY FOR DEVELOPERS
// ============================================================================

/*
OVERALL DATA FLOW:
1. CSV Loading: Robust CSV parser handles quoted fields and commas within values.
2. Data Inspection: Calculate missing values, display preview, create visualizations.
3. Preprocessing: 
   - Impute missing values (median for Age, mode for Embarked)
   - Standardize numeric features (Age, Fare)
   - One-hot encode categorical features (Sex, Pclass, Embarked)
   - Optional feature engineering (FamilySize, IsAlone)
4. Model Architecture:
   - Sigmoid Feature Gate layer for learnable feature importance
   - Hidden Dense layer (16 units, ReLU)
   - Output Dense layer (1 unit, Sigmoid)
5. Training:
   - 80/20 stratified split
   - Adam optimizer, binary crossentropy loss
   - Early stopping (patience=5)
   - Live visualization with tfjs-vis
6. Evaluation:
   - ROC curve and AUC calculation
   - Dynamic confusion matrix based on threshold
   - Precision, Recall, F1 metrics
   - Feature importance visualization from sigmoid gate
7. Prediction & Export:
   - Generate probabilities on test set
   - Apply threshold for binary predictions
   - Download submission.csv and probabilities.csv
   - Export trained model for reuse

KEY DESIGN PATTERNS:
- All data stays client-side, no server required
- Clear separation between data processing, model training, and UI updates
- Tensor disposal to prevent memory leaks
- Robust error handling with user-friendly messages
- Schema clearly marked for adaptation to other datasets

ADAPTING TO OTHER DATASETS:
1. Update TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ID_COLUMN in preprocessData()
2. Adjust model architecture if needed (input dimension changes automatically)
3. Modify preprocessing steps as needed for new features
4. Update visualizations to show relevant relationships

DEPLOYMENT:
- Works on GitHub Pages (static hosting)
- All dependencies loaded from CDN
- No build step required

*/
