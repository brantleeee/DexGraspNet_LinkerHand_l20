"""Reference-vs-reference outcomes, separate from policy and repeated worker counts."""


def summarize(rows):
    if not rows:raise ValueError('No reference trials')
    outcomes={k:0 for k in ['both_pass','refined_only','original_only','both_fail']};groups={}
    for r in rows:
        a,b=r['original_passed'],r['refined_passed']
        outcomes['both_pass' if a and b else 'refined_only' if b else 'original_only' if a else 'both_fail']+=1
        groups.setdefault(r['group'],[]).append(r)
    for group,items in groups.items():
        if len(items)!=6 or len({r['case'] for r in items})!=6:raise ValueError('Incomplete or duplicated six-direction group')
    return dict(trials_per_reference=len(rows),translation_samples=len(groups),outcomes=outcomes,
        original_passed=sum(r['original_passed'] for r in rows),refined_passed=sum(r['refined_passed'] for r in rows),
        original_all_six=sum(all(r['original_passed'] for r in v) for v in groups.values()),
        refined_all_six=sum(all(r['refined_passed'] for r in v) for v in groups.values()))
