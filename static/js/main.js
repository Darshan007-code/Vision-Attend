// VisionAttend Main Client Script

document.addEventListener('DOMContentLoaded', () => {
    // Auto-dismiss alert boxes after 6 seconds
    setTimeout(() => {
        const alerts = document.querySelectorAll('.animate-fade-in');
        alerts.forEach(el => {
            if (el.parentElement && el.parentElement.classList.contains('space-y-2')) {
                el.style.transition = 'opacity 0.5s ease';
                el.style.opacity = '0';
                setTimeout(() => el.remove(), 500);
            }
        });
    }, 6000);
});
