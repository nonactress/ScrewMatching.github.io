# -*- coding: utf-8 -*-
# badminton_scheduler_strict_gender_winrate.py
# 요구 보장:
# - 종목 하드 제약: 남복=남자만, 여복=여자만, 혼복=남1+여1
# - 예외: "안동회"는 여복에 한해 여자처럼 편성 가능
# - 라운드 내 중복 배정 금지
# - 출석 ≥ 16명이면 매 라운드 반드시 4경기 생성
# - 각 경기 옆에 예상 승률 출력

ROUNDS = 3
COURTS = 4
SLOTS_PER_ROUND = COURTS * 4  # 16명
D_ELO = 600  # 기대승률 스케일

# 등급 점수 맵핑(상위로 갈수록 간격 축소, S는 1000)
BASE_SCORES = {
    "F": 100,
    "E": 220,  # +120
    "D": 330,  # +110
    "C": 430,  # +100
    "B": 520,  # +90
    "A": 600,  # +80
    "S": 1000
}
GRADE_ORDER = {g:i for i,g in enumerate(["F","E","D","C","B","A","S"])}

ALPHA = 3.0  # 등급차 가중
BETA  = 1.0  # 강도차 가중

# 여복 예외 허용 대상(남성이지만 여복엔 들어갈 수 있음)
WOMEN_OVERRIDE = {"안동회"}

# 1) 전체 등록 명단
# 형식: (이름, 급수, 성별['M'/'F'])
players_master = [
     ("김동현","E","M"), ("김보연","F","F"), ("김예슬","F","F"),
     ("김하진","E","F"), ("부선민","D","M"), ("서동우","E","M"),
     ("신서연","D","F"),
     ("서현진","C","M"), ("송민준","S","M"), ("신서영","D","F"),
     ("안동회","F","M"), ("안승기","D","M"),
     ("양현준","C","M"), ("오승윤","D","M"),
     ("윤원준","E","M"), ("이우현","C","M"), ("이진서","C","M"),
     ("임희선","F","F"), ("전민규","D","M"), ("전민성","D","M"),
     ("정산별","E","M"), ("정영호","B","M"), ("정지윤","B","F"),
     ("조예성","D","F"), ("최유경","C","F"), ("최희수","E","F")
]

# 2) 오늘 출석자(비우면 전원 출석)
todays_present = {
    # "안서현","정지윤", ...
     "오승윤", "정지윤", "조예성", "부선민", "서동우", "서현진",
     "안승기", "정영호", "송민준", "김보연", "신서영", "이우현",
      "윤원준", "전민성", "김동현", "김하진", "안동회","최유경"
}

# ===== 내부 모델 =====
class Player:
    __slots__ = ("id","name","grade","gender","S","played_today")
    def __init__(self, pid, name, grade, gender):
        self.id = pid
        self.name = name
        self.grade = grade
        self.gender = gender
        self.S = BASE_SCORES.get(grade, 300)
        self.played_today = 0

def mk_players(master, present_set):
    roster = []
    pid = 0
    for n,g,s in master:
        if (not present_set) or (n in present_set):
            roster.append(Player(pid, n, g, s))
            pid += 1
    return roster

def ggap(a,b): return abs(GRADE_ORDER[a.grade] - GRADE_ORDER[b.grade])
def team_strength(pair, id2p): return id2p[pair[0]].S + id2p[pair[1]].S
def team_cost(a,b): return ALPHA * ggap(a,b) + BETA * abs(a.S - b.S)

def expected_prob(SA, SB, d=D_ELO):
    # 팀A가 이길 기대확률
    # E = 1 / (1 + 10^((SB - SA)/d))
    from math import pow
    return 1.0 / (1.0 + pow(10.0, (SB - SA) / d))

def compute_round_targets(fixed_pool):
    # 같은 출석 풀 기준으로 혼 목표를 라운드에 균등 분배, 남은 코트는 남/여로 분산
    males = sum(1 for p in fixed_pool if p.gender=="M")
    females = sum(1 for p in fixed_pool if p.gender=="F")
    # 여복 예외 남성은 여성으로 환산해 혼 가능량 추정에 반영하지 않음(혼은 M+F가 원칙)
    # 혼 총가능 대략:
    total_mix_possible = min(males, females) // 2
    base_mix = min(COURTS, total_mix_possible // ROUNDS)
    rem = min(COURTS, total_mix_possible) - base_mix * ROUNDS
    targets = []
    for r in range(ROUNDS):
        mix = base_mix + (1 if r < rem else 0)
        mix = min(mix, COURTS)
        rem_courts = COURTS - mix
        # 남/여 분배(대략)
        m_left_pairs = max(0, (males - 2*mix) // 2) // 2
        f_left_pairs = max(0, (females - 2*mix) // 2) // 2
        if m_left_pairs + f_left_pairs == 0:
            men = rem_courts; wom = 0
        else:
            ratio = m_left_pairs / (m_left_pairs + f_left_pairs)
            men = min(rem_courts, round(rem_courts * ratio))
            wom = rem_courts - men
        targets.append({"MIX":mix, "MEN":men, "WOM":wom})
    return targets

def select_round_pool(fixed_pool):
    # 오늘 적게 뛴 순으로 16명 선발
    sortedP = sorted(fixed_pool, key=lambda p: (p.played_today, p.gender))
    return sortedP[:SLOTS_PER_ROUND], sortedP[SLOTS_PER_ROUND:]

def split_by_gender_for_type(cands, ttype):
    # 종목에 맞춰 후보군을 분리 (여복 예외 반영)
    men = [p for p in cands if p.gender=="M"]
    wom = [p for p in cands if p.gender=="F"]
    if ttype == "WOM":
        # 여복엔 여성 + 예외 남성(여성처럼 취급) 포함
        wom_plus = wom + [p for p in cands if (p.gender=="M" and p.name in WOMEN_OVERRIDE)]
        return [], wom_plus
    return men, wom

def make_pairs(cands, ttype, max_gap=2, allow_extreme_S=False):
    # ttype: "MIX" / "MEN" / "WOM"
    men, wom = split_by_gender_for_type(cands, ttype)
    pairs = []
    if ttype == "MIX":
        # 반드시 남1+여1
        for m in men:
            for w in wom:
                # w가 여성이어야 함(여복 예외 남성은 혼복에선 여성으로 간주하지 않음)
                if w.gender != "F": 
                    continue
                if (max_gap is not None) and ggap(m,w) > max_gap: continue
                if not allow_extreme_S:
                    if ("S" in (m.grade,w.grade)) and (m.grade not in ("A","S") and w.grade not in ("A","S")):
                        continue
                pairs.append(((m.id,w.id), team_cost(m,w)))
    elif ttype == "MEN":
        same = men  # 남자만
        for i in range(len(same)):
            for j in range(i+1,len(same)):
                a,b = same[i], same[j]
                if (max_gap is not None) and ggap(a,b) > max_gap: continue
                if not allow_extreme_S:
                    if ("S" in (a.grade,b.grade)) and (a.grade not in ("A","S") and b.grade not in ("A","S")):
                        continue
                pairs.append(((a.id,b.id), team_cost(a,b)))
    else:  # WOM
        same = wom  # 여성 + 예외 남성
        for i in range(len(same)):
            for j in range(i+1,len(same)):
                a,b = same[i], same[j]
                # 혼복 제약 아님 → 예외 남성도 허용
                if (max_gap is not None) and ggap(a,b) > max_gap: continue
                if not allow_extreme_S:
                    if ("S" in (a.grade,b.grade)) and (a.grade not in ("A","S") and b.grade not in ("A","S")):
                        continue
                pairs.append(((a.id,b.id), team_cost(a,b)))
    pairs.sort(key=lambda x: x[1])
    return [p for p,_ in pairs]

def pair_to_matches(pairs, id2p, need, used_ids):
    # 팀 강도 차 최소화 + 라운드 내 중복 배정 금지
    avail_pairs = [p for p in pairs if (p[0] not in used_ids and p[1] not in used_ids)]
    scored = []
    for i in range(len(avail_pairs)):
        for j in range(i+1,len(avail_pairs)):
            A, B = avail_pairs[i], avail_pairs[j]
            if len({A[0],A[1],B[0],B[1]}) < 4: continue
            if any(pid in used_ids for pid in (A[0],A[1],B[0],B[1])): continue
            diff = abs(team_strength(A,id2p) - team_strength(B,id2p))
            scored.append(((i,j), diff))
    scored.sort(key=lambda x: x[1])
    matches = []
    taken = set()
    for (i,j), _ in scored:
        if len(matches) >= need: break
        if i in taken or j in taken: continue
        A, B = avail_pairs[i], avail_pairs[j]
        if len({A[0],A[1],B[0],B[1]}) < 4: continue
        if any(pid in used_ids for pid in (A[0],A[1],B[0],B[1])): continue
        matches.append((A,B))
        taken.add(i); taken.add(j)
        for pid in (A[0],A[1],B[0],B[1]): used_ids.add(pid)
    return matches

def force_fill(pool, used_ids, id2p, need_more):
    # 제약 완화하며 어떤 종목이든 매치 생성(성별 조합 원칙은 지킴 + 여복 예외)
    relax_plan = [
        (2, False, False),
        (3, False, False),
        (3, True,  False),
        (None, True, True),  # 마지막 단계: 등급 제약 무시. 단, 성별 조합 원칙은 유지 + 여복 예외 허용
    ]
    out = []
    for _ in range(need_more):
        made = False
        for max_gap, allow_S, ignore_all in relax_plan:
            rem = [p for p in pool if p.id not in used_ids]
            if len(rem) < 4: break
            all_pairs = []
            def gen_pairs(ttype):
                if ignore_all:
                    men, wom = split_by_gender_for_type(rem, ttype)
                    if ttype=="MIX":
                        # 혼은 남1+여1(여복 예외 남성은 여기선 여성 취급하지 않음)
                        true_wom = [w for w in wom if w.gender=="F"]
                        return [(m.id,w.id) for m in men for w in true_wom]
                    elif ttype=="MEN":
                        return [(men[i].id,men[j].id) for i in range(len(men)) for j in range(i+1,len(men))]
                    else:
                        # WOM: 여성 + 예외 남성
                        return [(wom[i].id,wom[j].id) for i in range(len(wom)) for j in range(i+1,len(wom))]
                else:
                    return make_pairs(rem, ttype, max_gap=max_gap, allow_extreme_S=allow_S)
            for t in ["MIX","MEN","WOM"]:
                for p in gen_pairs(t):
                    all_pairs.append((t,p))
            best = None
            for i in range(len(all_pairs)):
                for j in range(i+1,len(all_pairs)):
                    t1, A = all_pairs[i]
                    t2, B = all_pairs[j]
                    if len({A[0],A[1],B[0],B[1]}) < 4: continue
                    if any(pid in used_ids for pid in (A[0],A[1],B[0],B[1])): continue
                    diff = abs(team_strength(A,id2p) - team_strength(B,id2p))
                    tag = "혼" if ("MIX" in (t1,t2)) else ("남" if "MEN" in (t1,t2) else "여")
                    if (best is None) or (diff < best[0]):
                        best = (diff, tag, A, B)
            if best:
                _, tag, A, B = best
                out.append((tag, A, B))
                for pid in (A[0],A[1],B[0],B[1]): used_ids.add(pid)
                made = True
                break
        if not made:
            break
    return out

def brute_last_resort(pool, used_ids, id2p, need_more):
    # 최후 수단: 성별 조합 원칙만 지키며 강제로 채움(여복 예외 허용)
    out = []
    for _ in range(need_more):
        rem = [p for p in pool if p.id not in used_ids]
        if len(rem) < 4: break
        # 혼 후보(남+여(여성만))
        men = [p for p in rem if p.gender=="M"]
        wom_true = [p for p in rem if p.gender=="F"]
        # 남 후보
        men_pairs = [(men[i].id, men[j].id) for i in range(len(men)) for j in range(i+1,len(men))]
        # 여 후보(여성 + 예외 남성)
        wom_all = wom_true + [p for p in rem if p.gender=="M" and p.name in WOMEN_OVERRIDE]
        wom_pairs = [(wom_all[i].id, wom_all[j].id) for i in range(len(wom_all)) for j in range(i+1,len(wom_all))]
        mix_pairs = [(m.id, w.id) for m in men for w in wom_true]

        candidates = []
        for A in mix_pairs + men_pairs + wom_pairs:
            candidates.append(("혼" if A in mix_pairs else ("남" if A in men_pairs else "여"), A))

        best = None
        for ai in range(len(candidates)):
            tagA, A = candidates[ai]
            if A[0] in used_ids or A[1] in used_ids: continue
            for bi in range(ai+1,len(candidates)):
                tagB, B = candidates[bi]
                if B[0] in used_ids or B[1] in used_ids: continue
                if len({A[0],A[1],B[0],B[1]}) < 4: continue
                SA = team_strength(A,id2p); SB = team_strength(B,id2p)
                diff = abs(SA - SB)
                # 태그는 혼 우선, 아니면 동일 태그 우선
                tag = "혼" if ("혼" in (tagA, tagB)) else (tagA if tagA==tagB else tagA)
                if (best is None) or (diff < best[0]):
                    best = (diff, tag, A, B)
        if best is None:
            break
        _, tag, A, B = best
        out.append((tag, A, B))
        for pid in (A[0],A[1],B[0],B[1]): used_ids.add(pid)
    return out

def schedule(master, present_set):
    fixed_pool = mk_players(master, present_set)
    id2p = {p.id:p for p in fixed_pool}

    targets_all = compute_round_targets(fixed_pool)

    outputs = []
    for r in range(1, ROUNDS+1):
        # 1) 선발(오늘 적게 뛴 사람 우선)
        pool, _ = select_round_pool(fixed_pool)

        # 2) 목표치
        targets = targets_all[r-1]

        # 3) 팀 후보(기본 제약)
        mix_pairs = make_pairs(pool, "MIX", max_gap=2, allow_extreme_S=False)
        men_pairs = make_pairs(pool, "MEN", max_gap=2, allow_extreme_S=False)
        wom_pairs = make_pairs(pool, "WOM", max_gap=2, allow_extreme_S=False)

        used = set()
        matches = []

        def take(ttype, want):
            nonlocal matches
            if want <= 0: return
            src = {"MIX":mix_pairs,"MEN":men_pairs,"WOM":wom_pairs}[ttype]
            picked = pair_to_matches(src, id2p, want, used)
            tag = "혼" if ttype=="MIX" else ("남" if ttype=="MEN" else "여")
            for A,B in picked:
                matches.append((tag,A,B))

        take("MIX", targets["MIX"])
        take("MEN", targets["MEN"])
        take("WOM", targets["WOM"])

        # 4) 강제 채우기
        need_more = COURTS - len(matches)
        if need_more > 0:
            matches.extend(force_fill(pool, used, id2p, need_more))
        need_more = COURTS - len(matches)
        if len(fixed_pool) >= SLOTS_PER_ROUND and need_more > 0:
            matches.extend(brute_last_resort(pool, used, id2p, need_more))

        # 5) 플레이 카운트 반영
        played_ids = set()
        for _, A, B in matches:
            for pid in (A[0],A[1],B[0],B[1]):
                id2p[pid].played_today += 1
                played_ids.add(pid)

        # 6) 빠진사람(전체 출석 기준)
        wait_names = [p.name for p in fixed_pool if p.id not in played_ids]

        # 7) 출력(예상 승률 포함)
        lines = [f"{r}타임"]
        for i,(tag,A,B) in enumerate(matches, start=1):
            SA = team_strength(A,id2p); SB = team_strength(B,id2p)
            EA = expected_prob(SA, SB, D_ELO); EB = 1.0 - EA
            a1,a2 = id2p[A[0]].name, id2p[A[1]].name
            b1,b2 = id2p[B[0]].name, id2p[B[1]].name
            lines.append(
                f" {i}) {a1}, {a2} vs {b1}, {b2} [{tag}] "
                f"[승률 {round(EA*100):02d}% - {round(EB*100):02d}%]"
            )
        lines.append(f" 빠진사람({len(wait_names)}) ) " + (", ".join(wait_names) if wait_names else "없음"))

        # 8) 디버그 검증
        if len(fixed_pool) >= SLOTS_PER_ROUND and len(matches) < COURTS:
            lines.append(f" [경고] 인원 충분했는데 {COURTS}경기 미달. 로직 점검 필요.")
        if len(fixed_pool) >= SLOTS_PER_ROUND:
            expected_wait = len(fixed_pool) - SLOTS_PER_ROUND
            if len(wait_names) != expected_wait:
                lines.append(f" [경고] 빠진사람 수 불일치: 기대={expected_wait}, 실제={len(wait_names)}")

        outputs.append("\n".join(lines))

    return "\n\n".join(outputs)

if __name__ == "__main__":
    print(schedule(players_master, todays_present))
