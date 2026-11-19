/**
 * OSS Guard — UI Animations & Micro-Interactions
 * Vanilla JavaScript (ES6+) for smooth, controlled motion
 */

document.addEventListener('DOMContentLoaded', () => {
    initHeroParallax();
    // initCardTilt() desactivado: la interacción ahora es un hover CSS ultra suave y mínimo
    initMagneticButton();
});

/**
 * 1. Subtle 3D Parallax for the Hero Logo (.shield-art / .hero-shield-img)
 * Rotates smoothly in 3D space with a maximum angle of 8 degrees.
 */
function initHeroParallax() {
    const hero = document.querySelector('.hero');
    const logo = document.querySelector('.shield-art, .hero-shield-img');

    if (!hero || !logo) return;

    const MAX_TILT = 8; // degrees
    let rafId = null;

    hero.addEventListener('mousemove', (e) => {
        if (rafId) cancelAnimationFrame(rafId);

        rafId = requestAnimationFrame(() => {
            const rect = logo.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;

            // Normalized distance from logo center (-1 to 1)
            const heroRect = hero.getBoundingClientRect();
            const deltaX = (e.clientX - centerX) / (heroRect.width / 2);
            const deltaY = (e.clientY - centerY) / (heroRect.height / 2);

            // Clamp and scale to max tilt
            const clampedX = Math.max(-1, Math.min(1, deltaX));
            const clampedY = Math.max(-1, Math.min(1, deltaY));

            const rotateX = (-clampedY * MAX_TILT).toFixed(2);
            const rotateY = (clampedX * MAX_TILT).toFixed(2);

            logo.style.transition = 'transform 0.15s ease-out';
            logo.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
        });
    });

    hero.addEventListener('mouseleave', () => {
        if (rafId) cancelAnimationFrame(rafId);
        logo.style.transition = 'transform 0.6s cubic-bezier(0.16, 1, 0.3, 1)';
        logo.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg)';
    });
}

/**
 * 2. Subtle 3D Tilt for Result Cards (.result-card)
 * Tilts subtly on hover with a maximum angle of 4 degrees.
 */
function initCardTilt() {
    const cards = document.querySelectorAll('.result-card');
    if (!cards.length) return;

    // Ángulo ultra sutil (1 grado) para evitar mareo o desplazamiento excesivo
    const MAX_CARD_TILT = 1.0;

    cards.forEach((card) => {
        let cardRaf = null;
        let rect = null;

        // Medir dimensiones estables en mouseenter para evitar parpadeos en las esquinas
        card.addEventListener('mouseenter', () => {
            rect = card.getBoundingClientRect();
        });

        card.addEventListener('mousemove', (e) => {
            if (!rect) {
                rect = card.getBoundingClientRect();
            }

            if (cardRaf) cancelAnimationFrame(cardRaf);

            cardRaf = requestAnimationFrame(() => {
                if (!rect || rect.width <= 0 || rect.height <= 0) return;

                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                const centerX = rect.width / 2;
                const centerY = rect.height / 2;

                // Limitar entre -1 y 1 para un movimiento suave y predecible
                const deltaX = Math.max(-1, Math.min(1, (x - centerX) / centerX));
                const deltaY = Math.max(-1, Math.min(1, (y - centerY) / centerY));

                const rotateX = (-deltaY * MAX_CARD_TILT).toFixed(2);
                const rotateY = (deltaX * MAX_CARD_TILT).toFixed(2);

                // Elevación suave de 3px y rotación de apenas 1 grado
                card.style.transition = 'transform 0.28s cubic-bezier(0.16, 1, 0.3, 1)';
                card.style.transform = `perspective(1200px) translateY(-3px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
            });
        });

        card.addEventListener('mouseleave', () => {
            if (cardRaf) cancelAnimationFrame(cardRaf);
            rect = null;
            card.style.transition = 'transform 0.45s cubic-bezier(0.16, 1, 0.3, 1)';
            card.style.transform = 'perspective(1200px) translateY(0px) rotateX(0deg) rotateY(0deg)';
        });
    });
}

/**
 * 3. Magnetic Button Effect for .analyze-button
 * Gently attracts the button towards the cursor by up to 3px.
 */
function initMagneticButton() {
    const buttons = document.querySelectorAll('.analyze-button');
    if (!buttons.length) return;

    const MAX_DISPLACEMENT = 3; // pixels

    buttons.forEach((btn) => {
        let btnRaf = null;

        btn.addEventListener('mousemove', (e) => {
            if (btnRaf) cancelAnimationFrame(btnRaf);

            btnRaf = requestAnimationFrame(() => {
                const rect = btn.getBoundingClientRect();
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;

                const deltaX = (e.clientX - centerX) / (rect.width / 2);
                const deltaY = (e.clientY - centerY) / (rect.height / 2);

                const moveX = (Math.max(-1, Math.min(1, deltaX)) * MAX_DISPLACEMENT).toFixed(2);
                const moveY = (Math.max(-1, Math.min(1, deltaY)) * MAX_DISPLACEMENT).toFixed(2);

                btn.style.transition = 'transform 0.15s ease-out';
                btn.style.transform = `translate(${moveX}px, ${moveY}px)`;
            });
        });

        btn.addEventListener('mouseleave', () => {
            if (btnRaf) cancelAnimationFrame(btnRaf);
            btn.style.transition = 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1)';
            btn.style.transform = 'translate(0px, 0px)';
        });
    });
}
