class ParticleSystem {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.connections = [];
        this.mouse = { x: 0, y: 0 };
        this.resize();
        window.addEventListener('resize', () => this.resize());
        window.addEventListener('mousemove', (e) => {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
        });
        this.init();
        this.animate();
    }
    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }
    init() {
        const count = Math.min(80, Math.floor(window.innerWidth / 15));
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 2 + 0.5,
                opacity: Math.random() * 0.5 + 0.1,
                hue: Math.random() > 0.5 ? 190 : 260
            });
        }
    }
    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0 || p.x > this.canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = `hsla(${p.hue}, 100%, 70%, ${p.opacity})`;
            this.ctx.fill();
            for (let j = i + 1; j < this.particles.length; j++) {
                const p2 = this.particles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 150) {
                    const alpha = (1 - dist / 150) * 0.15;
                    this.ctx.beginPath();
                    this.ctx.moveTo(p.x, p.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.strokeStyle = `hsla(200, 100%, 70%, ${alpha})`;
                    this.ctx.lineWidth = 0.5;
                    this.ctx.stroke();
                }
            }
        }
        requestAnimationFrame(() => this.animate());
    }
}
const API = {
    async predict(sentence1, sentence2) {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sentence1, sentence2 })
        });
        if (!res.ok) throw new Error('Prediction failed');
        return await res.json();
    },
    async train(sentence1, sentence2, label) {
        const res = await fetch('/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sentence1, sentence2, label })
        });
        if (!res.ok) throw new Error('Training failed');
        return await res.json();
    },
    async getStats() {
        const res = await fetch('/stats');
        if (!res.ok) throw new Error('Failed to load stats');
        return await res.json();
    },
    async getLogs() {
        const res = await fetch('/logs');
        if (!res.ok) throw new Error('Failed to load logs');
        return await res.json();
    }
};
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(40px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
function animateGauge(score) {
    const gaugeEl = document.querySelector('.gauge-fill');
    const numberEl = document.querySelector('.gauge-number');
    if (!gaugeEl || !numberEl) return;
    const circumference = 628;
    const target = circumference - (score * circumference);
    gaugeEl.style.strokeDashoffset = target;
    let current = 0;
    const step = score / 60;
    const counter = setInterval(() => {
        current += step;
        if (current >= score) {
            current = score;
            clearInterval(counter);
        }
        numberEl.textContent = (current * 100).toFixed(1) + '%';
    }, 16);
}
function resetGauge() {
    const gaugeEl = document.querySelector('.gauge-fill');
    const numberEl = document.querySelector('.gauge-number');
    if (gaugeEl) gaugeEl.style.strokeDashoffset = 628;
    if (numberEl) numberEl.textContent = '0%';
}
async function handlePredict() {
    const s1 = document.getElementById('sentence1');
    const s2 = document.getElementById('sentence2');
    const btn = document.getElementById('predict-btn');
    const result = document.getElementById('predict-result');
    const interpEl = document.getElementById('result-interpretation');
    if (!s1 || !s2) return;
    const sentence1 = s1.value.trim();
    const sentence2 = s2.value.trim();
    if (!sentence1 || !sentence2) {
        showToast('Please enter both sentences', 'error');
        return;
    }
    btn.classList.add('btn-loading');
    resetGauge();
    try {
        const data = await API.predict(sentence1, sentence2);
        result.classList.add('visible');
        animateGauge(data.similarity);
        if (interpEl) interpEl.textContent = data.interpretation;
        showToast(`Similarity: ${data.percentage}%`, 'success');
    } catch (err) {
        showToast('Prediction failed. Is the server running?', 'error');
    } finally {
        btn.classList.remove('btn-loading');
    }
}
async function handleTrain() {
    const s1 = document.getElementById('train-sentence1');
    const s2 = document.getElementById('train-sentence2');
    const slider = document.getElementById('similarity-slider');
    const btn = document.getElementById('train-btn');
    if (!s1 || !s2 || !slider) return;
    const sentence1 = s1.value.trim();
    const sentence2 = s2.value.trim();
    const label = parseFloat(slider.value);
    if (!sentence1 || !sentence2) {
        showToast('Please enter both training sentences', 'error');
        return;
    }
    btn.classList.add('btn-loading');
    try {
        await API.train(sentence1, sentence2, label);
        showToast('Training submitted! Model is learning...', 'success');
        s1.value = '';
        s2.value = '';
        slider.value = 0.5;
        updateSliderValue();
        setTimeout(loadDashboard, 3000);
    } catch (err) {
        showToast('Training failed. Please try again.', 'error');
    } finally {
        btn.classList.remove('btn-loading');
    }
}
function updateSliderValue() {
    const slider = document.getElementById('similarity-slider');
    const display = document.getElementById('slider-value');
    if (slider && display) {
        display.textContent = parseFloat(slider.value).toFixed(2);
    }
}
async function loadStats() {
    try {
        const stats = await API.getStats();
        const el = (id) => document.getElementById(id);
        if (el('stat-model')) el('stat-model').textContent = stats.model_name || 'Sphinx V1';
        if (el('stat-params')) el('stat-params').textContent = stats.parameters_human || '—';
        if (el('stat-arch')) el('stat-arch').textContent = stats.architecture || 'Siamese Transformer';
        if (el('stat-sessions')) el('stat-sessions').textContent = stats.train_sessions || '0';
        if (el('stat-version')) el('stat-version').textContent = 'v' + (stats.version || '1.0.0');
        if (el('metric-params')) el('metric-params').textContent = stats.parameters_human || '—';
        if (el('metric-layers')) el('metric-layers').textContent = stats.num_layers || '—';
        if (el('metric-heads')) el('metric-heads').textContent = stats.num_heads || '—';
        if (el('metric-dmodel')) el('metric-dmodel').textContent = stats.d_model || '—';
        if (el('metric-vocab')) el('metric-vocab').textContent = (stats.vocab_size || 0).toLocaleString();
        if (el('metric-maxlen')) el('metric-maxlen').textContent = stats.max_length || '—';
        if (el('metric-sessions')) el('metric-sessions').textContent = stats.train_sessions || '0';
        if (el('metric-loss')) el('metric-loss').textContent = stats.loss_function || '—';
    } catch (err) {
        console.warn('Could not load stats:', err);
    }
}
async function loadDashboard() {
    try {
        const data = await API.getLogs();
        const tbody = document.getElementById('log-tbody');
        const empty = document.getElementById('log-empty');
        if (!tbody) return;
        tbody.innerHTML = '';
        if (!data.logs || data.logs.length === 0) {
            if (empty) empty.style.display = 'block';
            return;
        }
        if (empty) empty.style.display = 'none';
        const logs = data.logs.slice(-50).reverse();
        logs.forEach(log => {
            const row = document.createElement('tr');
            const match = log.match(/\[(.+?)\] TRAINED \| loss=([\d.]+) \| label=([\d.]+) \| s1="(.+?)" \| s2="(.+?)"/);
            if (match) {
                row.innerHTML = `
                    <td>${match[1]}</td>
                    <td>${parseFloat(match[2]).toFixed(4)}</td>
                    <td>${match[3]}</td>
                    <td title="${match[4]}">${match[4]}</td>
                `;
            } else {
                row.innerHTML = `<td colspan="4">${log}</td>`;
            }
            tbody.appendChild(row);
        });
        loadStats();
    } catch (err) {
        console.warn('Could not load logs:', err);
    }
}
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });
    document.querySelectorAll('.fade-in-up').forEach(el => {
        observer.observe(el);
    });
}
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const navHeight = document.querySelector('.nav').offsetHeight;
                const top = target.offsetTop - navHeight - 20;
                window.scrollTo({ top, behavior: 'smooth' });
            }
        });
    });
}
document.addEventListener('DOMContentLoaded', () => {
    function initTiltEffect() {
        const cards = document.querySelectorAll('.glass-card');
        cards.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                const rotateX = (y - centerY) / 10;
                const rotateY = (centerX - x) / 10;
                card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
            });
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
            });
        });
    }
    function initMouseParallax() {
        const hero = document.getElementById('hero');
        const heroContent = document.querySelector('.hero-content');
        if (!hero || !heroContent) return;
        hero.addEventListener('mousemove', (e) => {
            const x = (window.innerWidth - e.pageX * 2) / 100;
            const y = (window.innerHeight - e.pageY * 2) / 100;
            heroContent.style.transform = `translateX(${x}px) translateY(${y}px)`;
        });
    }
    new ParticleSystem('particle-canvas');
    initSmoothScroll();
    initScrollAnimations();
    initTiltEffect();
    initMouseParallax();
    loadStats();
    loadDashboard();
    const slider = document.getElementById('similarity-slider');
    if (slider) {
        slider.addEventListener('input', updateSliderValue);
    }
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePredict);
    }
    const trainBtn = document.getElementById('train-btn');
    if (trainBtn) {
        trainBtn.addEventListener('click', handleTrain);
    }
    document.querySelectorAll('#sentence1, #sentence2').forEach(el => {
        el.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                handlePredict();
            }
        });
    });
});
