async function analyze(){
  const q = userInput.value.trim();
  if(!q) return;

  appendMessage('user', q);
  userInput.value='';

  appendMessage('assistant','Thinking...');

  const resp = await fetch('/query', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({query:q, model:modelSelect.value})
  });

  const j = await resp.json();

  messages.lastElementChild.remove(); // remove typing message
  appendMessage('assistant', j.answer || 'No response');

  retrieved.innerHTML = '';
  (j.retrieved || []).forEach(it => {
    const block = document.createElement('div');
    block.innerHTML =
      `<strong>score ${it.score.toFixed(3)}</strong><br>` +
      `${it.type === 'incident' ? 'Incident ' + it.incident : it.source}<br>` +
      `<small>${it.text}</small>`;
    retrieved.appendChild(block);
  });
}

function appendMessage(role,text){
  const el = document.createElement('div');
  el.className = 'message ' + role;
  el.textContent = text;
  messages.appendChild(el);
  el.scrollIntoView({behavior:'smooth'});
}

async function ingest(){
  const pdfs = document.getElementById('pdfs').files;
  const excel = document.getElementById('excel').files[0];
  const folder = pdffolder.value || '';

  const form = new FormData();
  for(let f of pdfs) form.append('pdfs', f);
  if(excel) form.append('excel', excel);
  form.append('pdffolder', folder);

  ingestStatus.textContent = 'Uploading...';
  const res = await fetch('/ingest', {method:'POST',body:form});
  const j = await res.json();
  ingestStatus.textContent = j.status || 'Done';
}

async function clearIndex(){
  const res = await fetch('/clear_index',{method:'POST'});
  const j = await res.json();
  ingestStatus.textContent = j.status;
}

const userInput = document.getElementById('user-input');
const modelSelect = document.getElementById('model-select');
const messages = document.getElementById('messages');
const retrieved = document.getElementById('retrieved');
const ingestStatus = document.getElementById('ingest-status');
const pdffolder = document.getElementById('pdffolder');

window.addEventListener('load', ()=>{
  document.getElementById('send-btn').onclick = analyze;
  document.getElementById('ingest-btn').onclick = ingest;
  document.getElementById('clear-btn').onclick = clearIndex;

  modelSelect.innerHTML = `<option value="${window.DEFAULT_MODEL}">${window.DEFAULT_MODEL}</option>`;
});
