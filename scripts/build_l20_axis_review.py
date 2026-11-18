"""Add a no-WebGL per-point frozen-v5 normal selection page to a v6 viewer."""
import argparse,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--viewer',required=True,type=Path)
    p.add_argument('--source-library',type=Path,default=Path('data/hand_contacts/l20_contact_v5_20261117'));a=p.parse_args()
    frames={x['stable_id']:x for x in json.loads((a.source_library/'contact_frames.json').read_text())}
    source_hash=hashlib.sha256((a.source_library/'manifest.json').read_bytes()).hexdigest()
    page=(a.viewer/'middle_distal.html').read_text();encoded=page.split('<script id="scene-data" type="application/json">')[1].split('</script>')[0];data=json.loads(encoded)
    for pt in data['points']:
        ref=frames[pt['global_id']]
        assert pt['xyz_mm']==[float(x)*1000 for x in ref['origin_P_m']]
        pt['frame']['axes']=[ref[k] for k in ['Xc','Yc','Zc']]
    page=page.replace(encoded,json.dumps(data,separators=(',',':')))
    viewer_meta=json.loads((a.viewer/'manifest.json').read_text())
    current=json.loads((Path(viewer_meta['contact_library'])/'manifest.json').read_text())
    initial_rules=current.get('middle_red_rules',{})
    controls='''<h2>逐点选择法向（依据冻结v5）</h2><p>本页图形恢复为冻结v5三轴；下方表格保留当前库记录。先在点下拉框选择P号，再选轴方向。原红/绿/蓝三轴保持不变，黑箭头仅预览所选N。点“记录当前点”后可继续选其他点；最后下载JSON。已指定的中指规则会预填，未记录点保持待核查，下载不会自动改源库。</p><select id="normal-rule"><option value="Xc">旧红+ → N</option><option value="-Xc">旧红− → N</option><option value="Yc">旧绿+ → N</option><option value="-Yc">旧绿− → N</option><option value="Zc">旧蓝+ → N</option><option value="-Zc">旧蓝− → N</option></select><button id="record-normal">记录当前点</button><button id="clear-normal">取消当前点记录</button><button id="export-normals">下载中指方向JSON</button><pre id="review-status"></pre>'''
    page=page.replace('<p id="state"></p>',controls+'<p id="state"></p>')
    script='''
const rules=INITIAL_RULES, frozen=new Map(D.points.map(p=>[p.global_id,p.frame.axes.map(a=>a.slice())]));
const choose=document.getElementById('normal-rule'),reviewStatus=document.getElementById('review-status');
function previewRule(){
 for(const p of D.points)delete p.review_normal;
 const p=D.points.find(p=>String(p.global_id)===select.value);
 if(p){const r=choose.value,i=['Xc','Yc','Zc'].indexOf(r.replace('-','')),sign=r.startsWith('-')?-1:1;p.review_normal=frozen.get(p.global_id)[i].map(v=>v*sign);}
 reviewStatus.textContent=JSON.stringify(rules,null,2);request();
}
choose.addEventListener('change',previewRule);
select.addEventListener('change',()=>{choose.value=rules[select.value]||'Xc';previewRule();});
document.getElementById('record-normal').addEventListener('click',()=>{if(select.value==='all'){reviewStatus.textContent='请先选择单个P号';return;}rules[select.value]=choose.value;previewRule();});
document.getElementById('clear-normal').addEventListener('click',()=>{delete rules[select.value];previewRule();});
document.getElementById('export-normals').addEventListener('click',()=>{
 const body={source_manifest_sha256:SOURCE_HASH,middle_rules:rules};
 const url=URL.createObjectURL(new Blob([JSON.stringify(body,null,2)],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='l20_middle_v5_axis_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
'''.replace('SOURCE_HASH',json.dumps(source_hash)).replace('INITIAL_RULES',json.dumps(initial_rules))
    page+='<script>'+script+'</script>'
    (a.viewer/'middle_distal_axis_review.html').write_text(page)
    index=a.viewer/'index.html';index.write_text(index.read_text()+'<h2>中指待确认项</h2><a href="middle_distal_axis_review.html">逐点按冻结v5红/绿/蓝轴选择法向，并下载记录</a>')
    print(a.viewer/'middle_distal_axis_review.html')


if __name__=='__main__':main()
