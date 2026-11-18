"""Explicit paired initial-position samples and outcome statistics; no simulator imports."""
import numpy as np

CASES=['gx+','gx-','gy+','gy-','gz+','gz-']


def make_samples(episodes,num_envs,amplitude=.0005,seed=20261105):
    if episodes<1 or num_envs<6 or num_envs%6:raise ValueError('Positive episodes and a multiple of six environments required')
    if not np.isfinite(amplitude) or amplitude<0:raise ValueError('Noise must be finite and nonnegative')
    # A group has one local object translation, tested under all six gravities.
    groups=np.random.default_rng(seed).uniform(-amplitude,amplitude,(episodes,num_envs//6,3))
    return dict(schema=1,seed=seed,amplitude_m=amplitude,episodes=episodes,num_envs=num_envs,
        coordinate_frame='original_object_coordinates_before_scene_rotation',direction_order=CASES,
        offsets_m=np.repeat(groups,6,axis=1).tolist())


def validate_samples(samples,episodes,num_envs,amplitude):
    offsets=np.asarray(samples['offsets_m'],dtype=np.float64)
    if samples['schema']!=1 or samples['direction_order']!=CASES:raise ValueError('Unknown sample format or direction order')
    if samples['coordinate_frame']!='original_object_coordinates_before_scene_rotation':raise ValueError('Wrong sample coordinate frame')
    if samples['episodes']!=episodes or samples['num_envs']!=num_envs or samples['amplitude_m']!=amplitude:raise ValueError('Sample dimensions/bounds differ from CLI')
    if offsets.shape!=(episodes,num_envs,3) or not np.isfinite(offsets).all() or np.any(np.abs(offsets)>amplitude):raise ValueError('Invalid translation samples')
    grouped=offsets.reshape(episodes,num_envs//6,6,3)
    if not np.array_equal(grouped,np.repeat(grouped[:,:,:1,:],6,axis=2)):raise ValueError('Six directions must share each local translation')
    return offsets


class FixedUniform:
    """Pass one saved sample array through the existing env.reset interface."""
    def __init__(self,offsets,amplitude):self.offsets=np.array(offsets,copy=True);self.amplitude=amplitude
    def uniform(self,low,high,size):
        if low!=-self.amplitude or high!=self.amplitude or tuple(size)!=self.offsets.shape:raise ValueError('Environment reset requested unexpected noise')
        return self.offsets.copy()


def compare_initial_states(expected,actual):
    if expected.keys()!=actual.keys():raise ValueError('Initial-state fields differ')
    differences={}
    for key in expected:
        x=np.asarray(expected[key]);y=np.asarray(actual[key])
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError(f'Invalid initial state: {key}')
        differences[key]=float(np.max(np.abs(x-y)))
        if not np.array_equal(x,y):raise ValueError(f'Unpaired explicit initial state {key}: max difference={differences[key]}')
    return differences


def summarize_pairs(zero,policy):
    def keyed(rows):
        result={(r['episode'],r['env']):r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate trial identity')
        return result
    baseline=keyed(zero);learned=keyed(policy)
    if not baseline or baseline.keys()!=learned.keys():raise ValueError('Incomplete paired outcome coverage')
    pairs=[]
    for key,b in sorted(baseline.items()):
        p=learned[key]
        if b['case']!=p['case'] or b['group']!=p['group'] or b['offset_m']!=p['offset_m']:raise ValueError('Paired samples differ')
        pairs.append(dict(episode=key[0],env=key[1],group=b['group'],case=b['case'],
            zero_passed=b['passed'],policy_passed=p['passed'],
            outcome='both_pass' if b['passed'] and p['passed'] else 'policy_only' if p['passed'] else 'zero_only' if b['passed'] else 'both_fail'))
    grouped={}
    for row in pairs:grouped.setdefault(row['group'],[]).append(row)
    group_pairs=[]
    for group,rows in sorted(grouped.items()):
        if len(rows)!=6 or {r['case'] for r in rows}!=set(CASES):raise ValueError('Incomplete six-direction group')
        group_pairs.append(dict(group=group,zero_all_six=all(r['zero_passed'] for r in rows),policy_all_six=all(r['policy_passed'] for r in rows)))
    return dict(trials=len(pairs),six_direction_groups=len(group_pairs),
        outcomes={k:sum(p['outcome']==k for p in pairs) for k in ['both_pass','policy_only','zero_only','both_fail']},
        per_direction={c:dict(trials=sum(p['case']==c for p in pairs),
            zero_passed=sum(p['case']==c and p['zero_passed'] for p in pairs),
            policy_passed=sum(p['case']==c and p['policy_passed'] for p in pairs)) for c in CASES},
        groups=group_pairs,pairs=pairs)
