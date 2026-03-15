// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const fileNameSpan = document.getElementById('file-name');
    const textInput = document.getElementById('textInput');
    const checkBtn = document.getElementById('checkBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    const loadingState = document.getElementById('loadingState');
    const resultsSection = document.getElementById('resultsSection');
    
    const scoreText = document.getElementById('scoreText');
    const scoreTitle = document.getElementById('scoreTitle');
    const scoreSubtitle = document.getElementById('scoreSubtitle');
    const scoreCirclePath = document.getElementById('scoreCirclePath');
    const aiScoreText = document.getElementById('aiScoreText');
    const aiScoreTitle = document.getElementById('aiScoreTitle');
    const aiScoreSubtitle = document.getElementById('aiScoreSubtitle');
    const aiScoreCirclePath = document.getElementById('aiScoreCirclePath');
    
    const textDisplay = document.getElementById('textDisplay');
    const sentenceExplainContainer = document.getElementById('sentenceExplainContainer');
    const internalSourcesContainer = document.getElementById('internalSourcesContainer');
    const internetSourcesContainer = document.getElementById('internetSourcesContainer');
    const downloadPdfReportBtn = document.getElementById('downloadPdfReportBtn');

    function safeSetHTML(element, html) {
        if (element) {
            element.innerHTML = html;
        }
    }

    function safeAppend(parent, child) {
        if (parent) {
            parent.appendChild(child);
        }
    }

    let currentFile = null;
    let lastAnalysisData = null;
    let lastReportToken = null;

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            currentFile = e.target.files[0];
            fileNameSpan.textContent = currentFile.name;
            textInput.value = ''; // clear text if file selected
            textInput.disabled = true;
            fileNameSpan.style.color = 'var(--accent-color)';
        } else {
            resetFile();
        }
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
        textInput.value = '';
        textInput.disabled = false;
        resetFile();
        resultsSection.classList.add('hidden');
        lastAnalysisData = null;
        lastReportToken = null;
    });

    function onDownloadReportClicked(format) {
        if (!lastAnalysisData) {
            alert('Run an analysis first, then download the report.');
            return;
        }

        if (lastReportToken) {
            window.location.href = `/api/report/download?token=${encodeURIComponent(lastReportToken)}&format=${encodeURIComponent(format)}`;
            return;
        }

        // Compatibility fallback if token is not available.
        submitReportDownload(lastAnalysisData, format);
    }

    // Event delegation fallback in case the button is re-rendered or listener binding is missed.
    document.addEventListener('click', (event) => {
        const pdfTarget = event.target.closest('#downloadPdfReportBtn');
        if (pdfTarget) {
            onDownloadReportClicked('pdf');
        }
    });

    function submitReportDownload(reportData, format) {
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = '/api/report/download';
        form.target = '_blank';
        form.style.display = 'none';

        const payloadInput = document.createElement('input');
        payloadInput.type = 'hidden';
        payloadInput.name = 'analysis_json';
        payloadInput.value = JSON.stringify(reportData);
        form.appendChild(payloadInput);

        const formatInput = document.createElement('input');
        formatInput.type = 'hidden';
        formatInput.name = 'format';
        formatInput.value = format || 'html';
        form.appendChild(formatInput);

        document.body.appendChild(form);
        form.submit();
        document.body.removeChild(form);
    }

    async function runCheckRequest(formData) {
        const response = await fetch('/api/check', {
            method: 'POST',
            body: formData
        });

        const rawBody = await response.text();
        let parsed = null;
        try {
            parsed = rawBody ? JSON.parse(rawBody) : null;
        } catch (e) {
            parsed = null;
        }

        if (!response.ok) {
            const backendMessage = (parsed && (parsed.error || parsed.detail))
                ? (parsed.error || parsed.detail)
                : `Server responded with ${response.status}.`;
            throw new Error(backendMessage);
        }

        if (!parsed) {
            throw new Error('Backend returned an empty or invalid response.');
        }

        return parsed;
    }

    // Submitting for check
    checkBtn.addEventListener('click', async () => {
        const textValue = textInput.value.trim();
        
        if (!currentFile && !textValue) {
            alert('Please enter some text or upload a document to check.');
            return;
        }

        // Setup UI for loading
        loadingState.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        checkBtn.disabled = true;
        
        try {
            const formData = new FormData();
            if (currentFile) {
                formData.append('file', currentFile);
            } else {
                formData.append('text', textValue);
            }
            let data;
            try {
                data = await runCheckRequest(formData);
            } catch (firstError) {
                // One retry helps recover from transient local-server restarts.
                data = await runCheckRequest(formData);
            }
            
            if (data.error) {
                alert(data.error);
                loadingState.classList.add('hidden');
                checkBtn.disabled = false;
                return;
            }

            renderResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert(`An error occurred during analysis: ${error.message}`);
        } finally {
            loadingState.classList.add('hidden');
            checkBtn.disabled = false;
        }
    });

    function resetFile() {
        currentFile = null;
        fileInput.value = '';
        fileNameSpan.textContent = 'Upload PDF or DOCX file';
        fileNameSpan.style.color = 'inherit';
        textInput.disabled = false;
    }

    function renderResults(data) {
        if (!scoreText || !scoreCirclePath || !scoreTitle || !scoreSubtitle || !resultsSection) {
            throw new Error('UI is missing required score elements. Please refresh the page.');
        }

        lastAnalysisData = data;
        lastReportToken = data.report_token || null;

        // 1. Update Score Ring
        const score = data.plagiarism_score; // 0 to 100
        scoreText.textContent = `${Math.round(score)}%`;
        
        // svg dash array logic (100 is full circle)
        scoreCirclePath.setAttribute('stroke-dasharray', `${score}, 100`);
        
        // Color coding based on score
        let strokeColor = 'var(--success-color)';
        let judgment = 'Looks Good!';
        if (score > 20) {
            strokeColor = '#f59e0b'; // warning orange
            judgment = 'Some Similarities Found';
        }
        if (score > 50) {
            strokeColor = 'var(--danger-color)';
            judgment = 'High Plagiarism Detected!';
        }
        scoreCirclePath.style.stroke = strokeColor;
        scoreText.style.color = strokeColor;

        scoreTitle.textContent = judgment;
        const semanticSimilarity = Number(data.semantic_similarity_score || 0);
        scoreSubtitle.textContent = `${data.plagiarized_sentences} out of ${data.total_sentences} sentences match existing sources. Highest semantic similarity: ${Math.round(semanticSimilarity)}%.`;

        // 2. Update AI usage ring
        const aiScore = Number(data.ai_usage_score || 0);
        const aiLikelySentences = Number(data.ai_likely_sentences || 0);
        aiScoreText.textContent = `${Math.round(aiScore)}%`;
        aiScoreCirclePath.setAttribute('stroke-dasharray', `${aiScore}, 100`);

        let aiColor = 'var(--success-color)';
        let aiJudgment = 'Low AI Usage Probability';
        if (aiScore >= 40) {
            aiColor = '#f59e0b';
            aiJudgment = 'Moderate AI Usage Probability';
        }
        if (aiScore >= 70) {
            aiColor = 'var(--danger-color)';
            aiJudgment = 'High AI Usage Probability';
        }

        aiScoreCirclePath.style.stroke = aiColor;
        aiScoreText.style.color = aiColor;
        aiScoreTitle.textContent = aiJudgment;
        aiScoreSubtitle.textContent = `${aiLikelySentences} out of ${data.total_sentences} sentences show AI-like writing patterns.`;

        // 3. Render Text Highlighting
        safeSetHTML(textDisplay, '');
        safeSetHTML(sentenceExplainContainer, '');
        safeSetHTML(internalSourcesContainer, '');
        safeSetHTML(internetSourcesContainer, '');

        data.results.forEach((res, index) => {
            // Span for sentence
            const span = document.createElement('span');
            span.textContent = res.sentence + ' ';
            span.className = 'text-sentence';
            
            if (res.is_plagiarized) {
                span.classList.add('plagiarized');
                
                // Add sources to list
                res.sources.forEach(src => {
                    const simPercent = Math.round(src.similarity * 100);
                    const safeUrl = src.url === 'Internal Reference Database' ? '#' : src.url;
                    const sourceType = src.source_type === 'internal' ? 'Internal DB' : 'Internet';
                    const reasons = Array.isArray(res.flag_reasons) && res.flag_reasons.length > 0
                        ? res.flag_reasons.join('; ')
                        : 'Heuristic threshold triggered.';
                    const sentenceConfidence = (res.sentence_confidence || 'low').toLowerCase();
                    
                    const sourceHtml = `
                        <div class="source-card">
                            <div class="source-header">
                                <span class="sim-badge">${simPercent}% Match</span>
                            </div>
                            <div class="sentence-explain"><strong>Match Type:</strong> ${sourceType}</div>
                            <div class="matched-sentence">"${res.sentence}"</div>
                            <a href="${safeUrl}" target="_blank" class="source-url">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline; margin-right:4px;">
                                    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                                    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                                </svg>
                                ${src.url}
                            </a>
                            <div class="source-snippet">${src.snippet}</div>
                            <div class="sentence-explain"><strong>Why flagged:</strong> ${reasons}</div>
                            <div class="confidence-tag ${sentenceConfidence}">Confidence: ${sentenceConfidence.toUpperCase()}</div>
                        </div>
                    `;
                    if (src.source_type === 'internal' && internalSourcesContainer) {
                        internalSourcesContainer.insertAdjacentHTML('beforeend', sourceHtml);
                    } else if (internetSourcesContainer) {
                        internetSourcesContainer.insertAdjacentHTML('beforeend', sourceHtml);
                    }
                });
            }

            if (res.is_ai_likely) {
                span.classList.add('ai-likely');
            }

            if (res.is_plagiarized || res.is_ai_likely) {
                const reasons = Array.isArray(res.flag_reasons) && res.flag_reasons.length > 0
                    ? res.flag_reasons.join('; ')
                    : 'Heuristic threshold triggered.';
                const sentenceConfidence = (res.sentence_confidence || 'low').toLowerCase();
                const explainHtml = `
                    <div class="explain-card">
                        <div class="explain-title"><strong>Sentence:</strong> ${res.sentence}</div>
                        <div class="explain-reason"><strong>Why flagged:</strong> ${reasons}</div>
                        <div class="confidence-tag ${sentenceConfidence}">Confidence: ${sentenceConfidence.toUpperCase()}</div>
                    </div>
                `;
                if (sentenceExplainContainer) {
                    sentenceExplainContainer.insertAdjacentHTML('beforeend', explainHtml);
                }
            }
            
            safeAppend(textDisplay, span);
        });

        if (internalSourcesContainer && internalSourcesContainer.innerHTML === '') {
            internalSourcesContainer.innerHTML = '<p style="color: var(--text-secondary); text-align:center; padding: 1rem;">No internal matches detected.</p>';
        }

        if (internetSourcesContainer && internetSourcesContainer.innerHTML === '') {
            internetSourcesContainer.innerHTML = '<p style="color: var(--text-secondary); text-align:center; padding: 1rem;">No internet matches detected.</p>';
        }

        if (sentenceExplainContainer && sentenceExplainContainer.innerHTML === '') {
            sentenceExplainContainer.innerHTML = '<p style="color: var(--text-secondary); text-align:center; padding: 0.75rem;">No flagged sentences.</p>';
        }

        // Show results
        resultsSection.classList.remove('hidden');
        
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

});
