const params = new URLSearchParams(location.search), MOOD_KEY = 'lofi-mood';
// ?mood= wins, then the listener's own pick, then the clock (day runs 07:00 through 18:59).
function setMood() {
  const hour = new Date().getHours();
  let picked = null;
  try { picked = localStorage.getItem(MOOD_KEY); } catch {}
  const moods = ['day', 'night'];
  document.documentElement.dataset.mood = moods.includes(params.get('mood')) ? params.get('mood')
    : moods.includes(picked) ? picked : (hour >= 7 && hour < 19 ? 'day' : 'night');
}
setMood();
