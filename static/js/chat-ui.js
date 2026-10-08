/* Shared safe rendering: model output is always text, never injected HTML. */
window.ChatUI = {
  format(element, text) {
    element.replaceChildren();
    String(text || '').split(/\n\s*\n/).forEach(block => {
      const p = document.createElement('p');
      block.split(/(\*\*[^*]+\*\*)/g).forEach(part => {
        if (part.startsWith('**') && part.endsWith('**')) {
          const strong = document.createElement('strong'); strong.textContent = part.slice(2,-2); p.append(strong);
        } else p.append(document.createTextNode(part));
      });
      element.append(p);
    });
  },
  sources(items) {
    const details = document.createElement('details'); details.className = 'answer-sources';
    const summary = document.createElement('summary'); summary.textContent = 'What is this based on?'; details.append(summary);
    const intro = document.createElement('p');
    intro.textContent = items.length ? 'This answer is based on the following retrieved knowledge-base evidence:' : 'No source evidence was returned for this answer.';
    details.append(intro);
    items.forEach(item => {
      const article = document.createElement('article'), title = document.createElement('strong'), excerpt = document.createElement('p');
      title.textContent = item.title || 'Knowledge base'; excerpt.textContent = item.excerpt || '';
      article.append(title, excerpt); details.append(article);
    });
    return details;
  }
};
// Missing files show intentional labels; add the photographs without editing code.
document.querySelectorAll('img[data-photo]').forEach(img => {
  function failed() {
    img.hidden = true;
    const parent = img.parentElement;
    let fallback = parent.querySelector('.photo-fallback');
    if (!fallback) { fallback = document.createElement('span'); fallback.className = 'photo-fallback'; fallback.textContent = img.alt + ' · photo coming soon'; parent.append(fallback); }
    fallback.hidden = false;
  }
  img.addEventListener('error', failed);
  if (img.complete && !img.naturalWidth) failed();
});
const collegeSelect = document.getElementById('collegeSelect');
if (collegeSelect) {
  const chapters = [...document.querySelectorAll('.program-chapter')];
  function selectFromHash() {
    const hash = decodeURIComponent(location.hash.slice(1));
    const selected = chapters.find(c => c.id === hash || [...c.querySelectorAll('.program-story')].some(p => p.id === hash)) || chapters[0];
    chapters.forEach(c => c.hidden = c !== selected);
    collegeSelect.value = selected.id;
    const count = selected.querySelectorAll('.program-story').length;
    document.getElementById('programCount').textContent = count + (count === 1 ? ' program' : ' programs');
  }
  collegeSelect.addEventListener('change', () => { history.replaceState(null,'','#'+collegeSelect.value); selectFromHash(); });
  window.addEventListener('hashchange', selectFromHash); selectFromHash();
}
document.querySelectorAll('.program-gallery').forEach(gallery => {
  const slides = [...gallery.querySelectorAll('.gallery-slide')];
  const pause = gallery.querySelector('[data-pause]');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let index = 0, paused = motion.matches, visible = false;
  function show(n) { index = (n + slides.length) % slides.length; slides.forEach((s,i) => s.hidden = i !== index); gallery.querySelector('[data-counter]').textContent = `${index+1} / ${slides.length}`; }
  function label() { pause.textContent = paused ? 'Play' : 'Pause'; pause.setAttribute('aria-label', paused ? 'Play slideshow' : 'Pause slideshow'); }
  gallery.querySelector('[data-prev]').onclick = () => { paused = true; show(index-1); label(); };
  gallery.querySelector('[data-next]').onclick = () => { paused = true; show(index+1); label(); };
  pause.onclick = () => { paused = !paused; label(); };
  motion.addEventListener('change', () => { paused = motion.matches; label(); });
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; }).observe(gallery);
  setInterval(() => { if (visible && !paused && !document.hidden && !gallery.closest('[hidden]') && !gallery.matches(':hover') && !gallery.contains(document.activeElement)) show(index+1); }, 5500);
  label();
});
