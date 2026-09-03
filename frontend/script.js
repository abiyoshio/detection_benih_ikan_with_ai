document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnDetect = document.getElementById('btn-detect');
    const btnReset = document.getElementById('btn-reset');
    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    const totalCountEl = document.getElementById('total-count');
    const classCountsContainer = document.getElementById('class-counts-container');
    const resultImage = document.getElementById('result-image');

    let selectedFile = null;

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // Handle file input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            alert('Format tidak didukung. Harap unggah JPG atau PNG.');
            return;
        }

        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultSection.classList.add('hidden'); // Sembunyikan hasil sebelumnya jika ada
        };
        reader.readAsDataURL(file);
    }

    // Handle reset
    btnReset.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        resultSection.classList.add('hidden');
    });

    // Handle detection
    btnDetect.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Tampilkan loading
        btnDetect.disabled = true;
        loading.classList.remove('hidden');
        resultSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            // Memanggil API Backend (relative path, otomatis ke server yang sama)
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Terjadi kesalahan pada server');
            }

            // Tampilkan hasil
            displayResults(data);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Gagal melakukan deteksi: ' + error.message + '\n\nPastikan backend Flask sudah berjalan.');
        } finally {
            loading.classList.add('hidden');
            btnDetect.disabled = false;
        }
    });

    function setMetricBar(barId, valueId, value, unit = '%') {
        const bar = document.getElementById(barId);
        const val = document.getElementById(valueId);
        if (!bar || !val) return;

        // Set nilai teks
        val.textContent = value + unit;

        // Set warna bar berdasarkan nilai
        bar.classList.remove('low', 'medium', 'high');
        if (value >= 70) bar.classList.add('high');
        else if (value >= 40) bar.classList.add('medium');
        else bar.classList.add('low');

        // Animasi bar (delay kecil agar CSS transition aktif)
        setTimeout(() => { bar.style.width = Math.min(value, 100) + '%'; }, 50);
    }

    function displayResults(data) {
        // Tampilkan gambar hasil deteksi (dengan bounding box)
        if (data.result_image) {
            resultImage.src = data.result_image + '?t=' + Date.now(); // cache-bust
            resultImage.style.display = 'block';
        }

        // Set total hitungan
        totalCountEl.innerText = data.total_detected;

        // Clear previous class counts
        classCountsContainer.innerHTML = '';

        // Tampilkan breakdown per kelas
        if (Object.keys(data.counts).length > 0) {
            for (const [className, count] of Object.entries(data.counts)) {
                const card = document.createElement('div');
                card.className = 'class-card';
                card.innerHTML = `
                    <span class="name">${className}</span>
                    <span class="count">${count} ekor</span>
                `;
                classCountsContainer.appendChild(card);
            }
        } else {
            const emptyMsg = document.createElement('div');
            emptyMsg.style.gridColumn = '1 / -1';
            emptyMsg.style.textAlign = 'center';
            emptyMsg.style.color = 'var(--text-muted)';
            emptyMsg.innerText = 'Tidak ada bibit ikan yang terdeteksi dalam gambar ini.';
            classCountsContainer.appendChild(emptyMsg);
        }

        // --- Tampilkan statistik confidence gambar ini ---
        if (data.image_stats) {
            const stats = data.image_stats;
            // Reset bar avg-conf agar animasi berjalan ulang
            const barEl = document.getElementById('bar-avg-conf');
            if (barEl) {
                barEl.style.transition = 'none';
                barEl.style.width = '0%';
                barEl.getBoundingClientRect(); // trigger reflow
            }

            setMetricBar('bar-avg-conf', 'val-avg-conf', stats.avg_confidence);
        }

        // Tampilkan section hasil
        resultSection.classList.remove('hidden');
        
        // Scroll ke bawah agar hasil terlihat
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
});
