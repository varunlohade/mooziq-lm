const params = new URLSearchParams(location.search), MOOD_KEY = 'lofi-mood';
const systemTheme = matchMedia('(prefers-color-scheme: dark)');
function displayMood(mood) {
  document.documentElement.dataset.mood = mood;
  document.querySelector('meta[name="theme-color"]').content = mood === 'night' ? '#000000' : '#ffffff';
  document.querySelector('[data-theme-toggle]')?.setAttribute('aria-label', mood === 'night' ? 'Use light theme' : 'Use dark theme');
}
function setMood() {
  let picked = null;
  try { picked = localStorage.getItem(MOOD_KEY); } catch {}
  const moods = ['day', 'night'];
  displayMood(moods.includes(params.get('mood')) ? params.get('mood')
    : moods.includes(picked) ? picked : (systemTheme.matches ? 'night' : 'day'));
}
setMood();
