(() => {
'use strict';
const $ = id => document.getElementById(id);
let scenes = window.OC_WALKTHROUGH, index = 0, timer = null, live = false, result = null, busy = false;
const original = window.OC_WALKTHROUGH;
function txt(tag, text, cls) { const n = document.createElement(tag); n.textContent = text; if(cls)n.className=cls;return n; }
function node(x,y,w,title,sub,accent=false) {
 return `<g><rect x="${x}" y="${y}" width="${w}" height="76" rx="13" fill="${accent?'#eaf1ff':'#ffffff'}" stroke="${accent?'#0b3d91':'#9eb7d5'}" stroke-width="1.5"/><text x="${x+w/2}" y="${y+32}" fill="#132d4f" font-size="16" font-weight="650" text-anchor="middle">${title}</text><text x="${x+w/2}" y="${y+55}" fill="#34445c" font-size="12" text-anchor="middle">${sub}</text></g>`;
}
function edge(x,y,a,b){return `<path class="signal" d="M${x} ${y} L${a} ${b}" fill="none" stroke="#0b3d91" stroke-width="2" marker-end="url(#arrow)"/>`;}
function diagram(kind) {
 let art='';
 if(kind==='query'||kind==='prepare') art=node(25,45,210,'Student question','Original input',true)+edge(130,121,130,185)+node(25,190,210,'Retrieval query','Translate when needed')+edge(235,228,315,228)+node(325,190,210,'Ontology candidates','Semantic entity matching');
 else if(kind==='graph'||kind==='plan') art=node(20,35,210,'Ontology catalog','Entities + relationships')+edge(230,73,320,73)+node(330,35,210,'Proposed plan','Query or traversal',true)+edge(435,111,435,190)+node(330,200,210,'Validate + execute','Check schema and constraints',true)+edge(330,238,240,238)+node(20,200,210,'Graph evidence','Fallback flagged if needed');
 else if(kind==='text'||kind==='fusion') art=node(20,28,210,'Dense search','Semantic similarity',kind==='text')+node(330,28,210,'BM25 search','Keyword relevance',kind==='text')+edge(125,104,240,198)+edge(435,104,320,198)+node(175,204,210,'Rank fusion','1 / (60 + rank)',kind==='fusion');
 else if(kind==='context'||kind==='evidence') art=node(20,30,210,'Graph evidence','Recorded relationships')+node(330,30,210,'Text evidence','Selected source passages')+edge(125,106,240,194)+edge(435,106,320,194)+node(175,200,210,'Selected context','Keep source qualifiers',true);
 else if(kind==='answer') art=node(20,40,210,'Selected context','Evidence, not pretraining')+edge(230,78,320,78)+node(330,40,210,'Configured LLM','Grounding instructions',true)+edge(435,116,435,204)+node(330,210,210,'Returned answer','State gaps and conflicts');
 else if(kind==='error') art=node(175,120,210,'Run unavailable','Check model connection',true);
 else art=node(20,35,210,'Answer trace','Plan + evidence + notices',true)+edge(230,73,320,73)+node(330,35,210,'Inspect support','Verify source coverage')+edge(435,111,435,195)+node(330,205,210,'Evaluation','Measure faithfulness, etc.');
 return `<svg viewBox="0 0 560 340" role="img"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="#0b3d91"/></marker></defs>${art}</svg>`;
}
// Live content is built with DOM text nodes, never HTML from the model.
function excerpt(value, limit=240) {
 const clean=String(value ?? '').replace(/\s+/g,' ').trim();
 if(clean.length<=limit)return clean;
 const cut=clean.slice(0,limit);const boundary=cut.lastIndexOf(' ');
 return cut.slice(0,boundary>limit*.7?boundary:limit)+'…';
}
function evidenceCard(label, value, options={}) {
 const full=String(value ?? '');const card=txt('article','', 'live-card');
 card.append(txt('div',label,'card-label'));
 if(options.source)card.append(txt('div',options.source,'card-source'));
 card.append(txt('p',options.exact?full:excerpt(full),'card-excerpt'));
 if(!options.exact && full.replace(/\s+/g,' ').trim().length>240){
  const expand=txt('details','', 'card-expand');expand.append(txt('summary','Read full text'),txt('pre',full));
  expand.addEventListener('toggle',()=>{if(expand.open)stop();});card.append(expand);
 }
 if(options.score)card.append(txt('small',options.score,'card-score'));
 return card;
}
function renderLive(s){
 const host=$('stage-art');host.replaceChildren();host.classList.add('live-art');host.removeAttribute('aria-hidden');
 const flow=txt('div','', 'live-flow');host.append(flow);
 const question=result?.trace?.query || result?.steps?.find(x=>x.type==='query')?.text || '';
 if(question && s.type!=='query')flow.append(evidenceCard('Your question',question,{exact:true}));
 function arrow(){const a=txt('div','↓','flow-arrow');a.setAttribute('aria-hidden','true');flow.append(a);}
 if(s.type==='query'){
  flow.append(evidenceCard(s.text?'Your exact question':'Ask a question to begin',s.text||s.narration,{exact:true}));
 }else if(s.type==='plan'){
  if(question)arrow();
  const plans=[];const data=s.data||{};
  (data.queries||[]).forEach((query,i)=>{
   const lines=[];
   (query.patterns||[]).forEach(p=>lines.push(Array.isArray(p)?p.join(' → '):JSON.stringify(p)));
   if(query.select?.length)lines.push('Return: '+query.select.join(', '));
   if(query.filters?.length)lines.push('Filters: '+JSON.stringify(query.filters));
   plans.push({label:`Recorded query ${i+1}`,text:lines.join('\n')||JSON.stringify(query)});
  });
  (data.traversals||[]).forEach((t,i)=>plans.push({label:`Recorded traversal ${i+1}`,text:JSON.stringify(t,null,2)}));
  if(!plans.length)plans.push({label:'Graph plan',text:Object.keys(data).length?JSON.stringify(data,null,2):'No graph plan was recorded for this run.'});
  plans.forEach(p=>flow.append(evidenceCard(p.label,p.text)));
  flow.append(txt('small','These are plan patterns, not proof that each relationship was found. The next evidence stage shows the recorded results.','flow-note'));
 }else if(s.items){
  if(question)arrow();
  if(s.fallback)flow.append(txt('p','Approximate matches · not an exhaustive result','fallback-notice'));
  flow.append(txt('div',`${s.items.length} evidence block${s.items.length===1?'':'s'} · shortened verbatim excerpts`,'card-label'));
  const cards=txt('div','', 'evidence-cards');flow.append(cards);
  function add(item,i,target){target.append(evidenceCard(`Evidence ${i+1}`,item.text,{source:item.title||'Recorded evidence',score:typeof item.score==='number'?`${item.score_label||'Recorded score'}: ${item.score.toFixed(5)}`:null}));}
  s.items.slice(0,3).forEach((item,i)=>add(item,i,cards));
  if(s.items.length>3){const more=txt('details','', 'more-evidence');more.append(txt('summary',`Show ${s.items.length-3} more evidence blocks`));s.items.slice(3).forEach((item,i)=>add(item,i+3,more));more.addEventListener('toggle',()=>{if(more.open)stop();});flow.append(more);}
  if(!s.items.length)flow.append(evidenceCard('No evidence recorded','This stage returned no evidence blocks.'));
 }else if(s.type==='answer'){
  const contexts=result?.trace?.selected_contexts||[];
  if(question)arrow();
  if(contexts.length){flow.append(evidenceCard('Supporting context',contexts[0],{source:`Excerpt from context 1 of ${contexts.length}`}));arrow();}
  flow.append(evidenceCard('Actual returned answer',s.text||'No answer was returned.'));
 }else{
  if(question)arrow();
  flow.append(evidenceCard(s.stage,s.text||s.narration||'No content recorded.',{exact:s.stage==='Retrieval question'}));
 }
}

function stop(){clearInterval(timer);timer=null;$('play').textContent='Play';document.body.classList.add('paused');}
function render(){
 const s=scenes[index]; $('scene-kicker').textContent=`${live?'RECORDED RUN':'HOW IT WORKS'} / STEP ${String(index+1).padStart(2,'0')}`;
 $('scene-title').textContent=s.stage;$('scene-description').textContent=s.narration || '';
 $('scene-detail').textContent=live?'The cards show this run’s recorded content. Expand a card to read the full text.':s.detail;
 if(live)renderLive(s);else{$('stage-art').classList.remove('live-art');$('stage-art').setAttribute('aria-hidden','true');$('stage-art').innerHTML=diagram(s.kind||s.type);}
 $('step-count').textContent=`${index+1} / ${scenes.length}`;
 $('previous').disabled=index===0||busy;$('next').disabled=index===scenes.length-1||busy;
 $('timeline').replaceChildren(...scenes.map((s,i)=>{const b=txt('button',s.short||s.stage);b.prepend(txt('small',String(i+1).padStart(2,'0')));b.setAttribute('aria-label',`Step ${i+1}: ${s.stage}`);if(i===index)b.setAttribute('aria-current','step');b.disabled=busy;b.onclick=()=>{stop();index=i;render();};return b;}));
 const content=$('evidence-content');content.replaceChildren();$('evidence').hidden=!live;
 if(live){if(s.data)content.append(txt('pre',JSON.stringify(s.data,null,2)));if(s.items)s.items.forEach(item=>{const article=document.createElement('article');article.append(txt('strong',item.title),txt('pre',item.text));if(typeof item.score==='number')article.append(txt('p',`${item.score_label}: ${item.score.toFixed(5)}`));content.append(article);});if(!content.childNodes.length)content.append(txt('pre',s.text||s.narration||'No evidence recorded for this stage.'));}
}
function play(){if(busy)return;if(timer){stop();return;}if(index===scenes.length-1)index=0;render();document.body.classList.remove('paused');$('play').textContent='Pause';timer=setInterval(()=>{if(index<scenes.length-1){index++;render();}else stop();},Number($('pace').value));}
function setView(isLive){if(busy)return;stop();live=isLive;$('query-panel').hidden=!live;$('live-btn').setAttribute('aria-pressed',String(live));$('demo-btn').setAttribute('aria-pressed',String(!live));index=0;scenes=live?(result?.steps||[{stage:'Ready for a real question',type:'query',narration:'Enter a question above. The completed response supplies the trace used in this presentation.'}]):original;$('trace-badge').textContent=live?(result?.trace_kind==='executed'?'EXECUTED TRACE · REPLAY':'LIVE MODE · NO SUCCESSFUL RUN YET'):'ILLUSTRATION · NOT A LIVE RUN';$('download').hidden=!live||!result;$('notice').textContent=live?'Replay order is an explanation of recorded stages, not real-time execution telemetry. Unrecorded candidate rankings and intermediate timings are not simulated.':'The walkthrough explains the supplied oc-rag-3.6 pipeline. Diagram nodes are illustrative, not actual retrieved campus facts.';render();}
$('demo-btn').onclick=()=>setView(false);$('live-btn').onclick=()=>setView(true);$('play').onclick=play;$('previous').onclick=()=>{stop();index=Math.max(0,index-1);render();};$('next').onclick=()=>{stop();index=Math.min(scenes.length-1,index+1);render();};$('replay').onclick=()=>{stop();index=0;render();};$('pace').onchange=()=>{if(timer){stop();play();}};
$('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}catch{$('notice').textContent='Fullscreen is unavailable. You can use your browser’s fullscreen command.';}};
$('query-form').onsubmit=async e=>{e.preventDefault();if(busy)return;const question=$('question').value.trim();if(!question)return;stop();result=null;scenes=[{stage:'Retrieving your evidence',type:'query',text:question,narration:'The backend is processing this question. Its recorded evidence will appear when the run completes.'}];index=0;$('trace-badge').textContent='RUNNING · WAITING FOR TRACE';$('download').hidden=true;busy=true;$('run-btn').disabled=true;$('run-btn').textContent='Retrieving…';$('demo-btn').disabled=$('live-btn').disabled=true;$('play').disabled=$('replay').disabled=true;render();$('notice').textContent='Running retrieval and generation. The completed trace will appear here; model speed determines the wait.';
 try{const response=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,modes:[$('mode').value]})});const data=await response.json();if(!response.ok||data.error)throw Error(data.error||'Request failed');if(!data.results?.[0]?.steps?.length)throw Error('The server returned no trace.');result=data.results[0];}
 catch(err){result=null;scenes=[{stage:'Run failed',type:'error',narration:'No completed trace is available. Please retry the query.'}];index=0;$('trace-badge').textContent='RUN FAILED · NO TRACE';$('notice').textContent=`Could not complete the run: ${err.message}`;}
 finally{busy=false;$('run-btn').disabled=false;$('run-btn').textContent='Run retrieval';$('demo-btn').disabled=$('live-btn').disabled=false;$('play').disabled=$('replay').disabled=false;if(result)setView(true);else render();}
};
$('download').onclick=()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='oc-rag-executed-trace.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON','SUMMARY'].includes(document.activeElement?.tagName)||busy)return;if(e.code==='Space'){e.preventDefault();play();}if(e.key==='ArrowRight')$('next').click();if(e.key==='ArrowLeft')$('previous').click();});
async function boot(){try{const res=await fetch('/api/status');if(!res.ok)throw Error();const data=await res.json();const info=data.model_info||{};$('model-status').textContent=`${data.llm_enabled?'Configured model':'Model unavailable'}: ${info.model||'Not reported'} · ${info.pipeline||'Pipeline version not reported'}`;$('mode').replaceChildren(...(data.modes||[]).map(mode=>{const o=txt('option',data.mode_meta?.[mode]?.title||mode);o.value=mode;return o;}));$('mode').value='ontology_contextual_rag';}catch{$('model-status').textContent='Live backend unavailable. The illustrated walkthrough still works.';}}
setView(true);boot();
})();
