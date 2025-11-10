// static/js/chatbot.js
document.getElementById('askBtn').addEventListener('click', async () => {
  const q = document.getElementById('question').value;
  if (!q) return alert('Type a question.');
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({text: q})
  });
  const data = await res.json();
  document.getElementById('reply').innerText = data.reply;
});
