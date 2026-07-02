import json
import os
import re


# ===================== SIMULADORES DETERMINISTAS =====================
# Solo VALIDAN el plan (no lo resuelven). Validados 574/574 contra Examples.json.

class _BlocksWorld:
    def __init__(self, stmt):
        init, goal = re.split(r"My goal is to", stmt, maxsplit=1)
        self.on = {}
        self.holding = None
        for x, y in re.findall(r"the (\w+) block is on top of the (\w+) block", init, re.I):
            self.on[x] = y
        for x in re.findall(r"the (\w+) block is on the table", init, re.I):
            self.on[x] = "table"
        m = re.search(r"holding the (\w+) block|the (\w+) block is being held", init, re.I)
        if m:
            self.holding = m.group(1) or m.group(2)
        self.goal_on = {}
        for x, y in re.findall(r"the (\w+) block is on top of the (\w+) block", goal, re.I):
            self.goal_on[x] = y
        for x in re.findall(r"the (\w+) block is on the table", goal, re.I):
            self.goal_on[x] = "table"

    def _clear(self, b):
        return b != self.holding and all(v != b for v in self.on.values())

    def apply(self, verb, args):
        if verb == "engage_payload":
            (x,) = args
            if self.holding is not None: return f"la mano no esta vacia (sostiene {self.holding})"
            if self.on.get(x) != "table": return f"{x} no esta en la mesa"
            if not self._clear(x): return f"{x} tiene algo encima"
            del self.on[x]; self.holding = x
        elif verb == "release_payload":
            (x,) = args
            if self.holding != x: return f"no sostiene {x}"
            self.on[x] = "table"; self.holding = None
        elif verb == "mount_node":
            x, y = args
            if self.holding != x: return f"no sostiene {x}"
            if not self._clear(y): return f"{y} no esta libre"
            self.on[x] = y; self.holding = None
        elif verb == "unmount_node":
            x, y = args
            if self.on.get(x) != y: return f"{x} no esta sobre {y}"
            if not self._clear(x): return f"{x} tiene algo encima"
            if self.holding is not None: return "la mano no esta vacia"
            del self.on[x]; self.holding = x
        else:
            return f"accion desconocida {verb}"
        return None

    def goal_reached(self):
        return all(self.on.get(x) == y for x, y in self.goal_on.items())


class _Mystery:
    def __init__(self, stmt):
        init, goal = re.split(r"My goal is to", stmt, maxsplit=1)
        self.craves = set(re.findall(r"object (\w+) craves object (\w+)", init, re.I))
        self.province = set(re.findall(r"province object (\w+)", init, re.I))
        self.planet = set(re.findall(r"planet object (\w+)", init, re.I))
        self.pain = set(re.findall(r"pain object (\w+)", init, re.I))
        self.harmony = bool(re.search(r"\bharmony\b", init, re.I))
        self.goal_craves = set(re.findall(r"object (\w+) craves object (\w+)", goal, re.I))

    def apply(self, verb, args):
        if verb == "attack":
            (x,) = args
            if x not in self.province: return f"no province({x})"
            if x not in self.planet: return f"no planet({x})"
            if not self.harmony: return "no harmony"
            self.pain.add(x); self.province.discard(x); self.planet.discard(x); self.harmony = False
        elif verb == "succumb":
            (x,) = args
            if x not in self.pain: return f"no pain({x})"
            self.province.add(x); self.planet.add(x); self.harmony = True; self.pain.discard(x)
        elif verb == "overcome":
            x, y = args
            if y not in self.province: return f"no province({y})"
            if x not in self.pain: return f"no pain({x})"
            self.harmony = True; self.province.add(x); self.craves.add((x, y))
            self.province.discard(y); self.pain.discard(x)
        elif verb == "feast":
            x, y = args
            if (x, y) not in self.craves: return f"no craves({x},{y})"
            if x not in self.province: return f"no province({x})"
            if not self.harmony: return "no harmony"
            self.pain.add(x); self.province.add(y)
            self.craves.discard((x, y)); self.province.discard(x); self.harmony = False
        else:
            return f"accion desconocida {verb}"
        return None

    def goal_reached(self):
        return self.goal_craves <= self.craves


class AssemblyAgent:
    """
    Resuelve escenarios PlanBench (Blocksworld ofuscado + Mystery) con Qwen3-8B
    en modo COMPLETION determinista, usando FEW-SHOT dirigido por dominio.

    Cada `scenario_context` ya trae 1 ejemplo resuelto. Este agente antepone
    K ejemplos adicionales del MISMO dominio (tomados de Examples.json, seleccion
    determinista) para que Qwen imite mejor el patron de plan optimo. Luego parsea
    la salida NL de Qwen a la forma canonica con parentesis del evaluador.
    """

    SYSTEM_PROMPT = (
        "You are an expert automated-assembly planner. Complete the plan that "
        "comes after the final [PLAN] tag with the SHORTEST valid sequence of "
        "actions that reaches the goal, copying EXACTLY the wording and format of "
        "the worked examples. Read the goal carefully: 'A on top of B' means A must "
        "end up directly above B (not B above A), and every action must respect its "
        "preconditions. Output only the action lines, one per line, and finish with "
        "[PLAN END]. Do not add explanations."
    )

    # ----- Patrones NL -> canonico -----------------------------------------
    _BW = [
        (re.compile(r"unmount_node\s+the\s+(\w+)\s+block\s+from\s+on\s+top\s+of\s+the\s+(\w+)\s+block", re.I),
         lambda m: f"(unmount_node {m.group(1).lower()} {m.group(2).lower()})"),
        (re.compile(r"(?:un[\s_]?stack)\s+the\s+(\w+)\s+block\s+from\s+(?:on\s+top\s+of\s+)?the\s+(\w+)\s+block", re.I),
         lambda m: f"(unmount_node {m.group(1).lower()} {m.group(2).lower()})"),
        (re.compile(r"mount_node\s+the\s+(\w+)\s+block\s+on\s+top\s+of\s+the\s+(\w+)\s+block", re.I),
         lambda m: f"(mount_node {m.group(1).lower()} {m.group(2).lower()})"),
        (re.compile(r"stack\s+the\s+(\w+)\s+block\s+on\s+top\s+of\s+the\s+(\w+)\s+block", re.I),
         lambda m: f"(mount_node {m.group(1).lower()} {m.group(2).lower()})"),
        (re.compile(r"pick\s+up\s+the\s+(\w+)\s+block", re.I),
         lambda m: f"(engage_payload {m.group(1).lower()})"),
        (re.compile(r"put\s+down\s+the\s+(\w+)\s+block", re.I),
         lambda m: f"(release_payload {m.group(1).lower()})"),
    ]
    _MY = [
        (re.compile(r"feast\s+(?:object\s+)?(\w+)\s+from\s+(?:object\s+)?(\w+)", re.I),
         lambda m: f"(feast {m.group(1).lower()} {m.group(2).lower()})"),
        (re.compile(r"overcome\s+(?:object\s+)?(\w+)\s+from\s+(?:object\s+)?(\w+)", re.I),
         lambda m: f"(overcome {m.group(1).lower()} {m.group(2).lower()})"),
        (re.compile(r"attack\s+(?:object\s+)?(\w+)", re.I),
         lambda m: f"(attack {m.group(1).lower()})"),
        (re.compile(r"succumb\s+(?:object\s+)?(\w+)", re.I),
         lambda m: f"(succumb {m.group(1).lower()})"),
    ]

    def __init__(self, few_shot_k: int = 4, max_repairs: int = 1,
                 examples_path: str = "Examples.json"):
        self.system_prompt = self.SYSTEM_PROMPT
        self.few_shot_k = few_shot_k
        self.max_repairs = max_repairs
        # Pools de ejemplos (statement + plan NL) por dominio, ordenados de forma
        # determinista. Si no hay Examples.json, degradamos a 1-shot (el que ya
        # trae el propio escenario).
        self._pools = {"blocks": [], "mystery": []}
        if few_shot_k > 0 and os.path.exists(examples_path):
            self._build_pools(examples_path)

    # ---------- construccion de pools de few-shot ----------
    @staticmethod
    def _domain(sc: str) -> str:
        return "mystery" if "block" not in sc.lower() else "blocks"

    @staticmethod
    def _final_statement(sc: str) -> str:
        idx = sc.rfind("[STATEMENT]")
        tail = sc[idx:] if idx != -1 else sc
        stmt = re.split(r"My plan is as follows:", tail)[0]
        return stmt.replace("[STATEMENT]", "").strip()

    @classmethod
    def _num_objects(cls, stmt: str, domain: str) -> int:
        """Nº de bloques/objetos distintos en el enunciado (para emparejar tamano)."""
        if domain == "mystery":
            return len(set(re.findall(r"object\s+(\w+)", stmt, re.I)))
        return len(set(re.findall(r"(\w+)\s+block", stmt, re.I)))

    @staticmethod
    def _canon_to_nl(action: str) -> str:
        p = action.strip("() ").split()
        v = p[0]
        if v == "engage_payload":  return f"pick up the {p[1]} block"
        if v == "release_payload": return f"put down the {p[1]} block"
        if v == "mount_node":      return f"mount_node the {p[1]} block on top of the {p[2]} block"
        if v == "unmount_node":    return f"unmount_node the {p[1]} block from on top of the {p[2]} block"
        if v == "attack":          return f"attack object {p[1]}"
        if v == "succumb":         return f"succumb object {p[1]}"
        if v == "feast":           return f"feast object {p[1]} from object {p[2]}"
        if v == "overcome":        return f"overcome object {p[1]} from object {p[2]}"
        return action

    def _build_pools(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        by_dom = {"blocks": [], "mystery": []}
        for e in data:
            sc = e["scenario_context"]
            dom = self._domain(sc)
            nl = "\n".join(self._canon_to_nl(a) for a in e["target_action_sequence"])
            stmt = self._final_statement(sc)
            by_dom[dom].append({
                "id": e["assembly_task_id"],
                "stmt": stmt,
                "nl": nl,
                "cx": e["complexity_level"],
                "n": self._num_objects(stmt, dom),
            })
        # Orden determinista por (complejidad, id).
        for dom, items in by_dom.items():
            items.sort(key=lambda s: (s["cx"], s["id"]))
            self._pools[dom] = items

    @staticmethod
    def _spread(items, k):
        """k posiciones equiespaciadas -> diversidad de complejidad, determinista."""
        if k <= 0 or not items:
            return []
        step = max(1, len(items) // k)
        return [items[min(i * step, len(items) - 1)] for i in range(k)]

    def _pick_shots(self, domain: str, exclude_stmt: str, target_n: int):
        pool = [s for s in self._pools[domain] if s["stmt"] != exclude_stmt]
        k = min(self.few_shot_k, len(pool))
        if k == 0:
            return []
        # 1) preferimos ejemplos con el MISMO nº de objetos que el problema.
        same = [s for s in pool if s["n"] == target_n]
        shots = self._spread(same, k)
        # 2) si no alcanzan, completamos con los mas cercanos en tamano.
        if len(shots) < k:
            chosen = {s["id"] for s in shots}
            rest = sorted((s for s in pool if s["id"] not in chosen),
                          key=lambda s: (abs(s["n"] - target_n), s["cx"], s["id"]))
            shots += rest[: k - len(shots)]
        return shots

    def _make_shot_block(self, stmt: str, nl_plan: str) -> str:
        return (f"[STATEMENT]\n{stmt}\nMy plan is as follows:\n\n"
                f"[PLAN]\n{nl_plan}\n[PLAN END]\n\n")

    def build_prompt(self, scenario_context: str) -> str:
        """Inserta K ejemplos del mismo dominio antes del problema final."""
        domain = self._domain(scenario_context)
        stmt = self._final_statement(scenario_context)
        shots = self._pick_shots(domain, stmt, self._num_objects(stmt, domain))
        if not shots:
            return scenario_context
        idx = scenario_context.rfind("[STATEMENT]")
        prefix, final_block = scenario_context[:idx], scenario_context[idx:]
        inserted = "".join(self._make_shot_block(s["stmt"], s["nl"]) for s in shots)
        return prefix + inserted + final_block

    # ---------- parsing ----------
    def _patterns_for(self, sc: str):
        return self._MY if self._domain(sc) == "mystery" else self._BW

    def _parse_plan(self, raw: str, scenario_context: str) -> list:
        patterns = self._patterns_for(scenario_context)
        actions = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if "PLAN END" in line.upper():
                break
            for rgx, fmt in patterns:
                m = rgx.search(line)
                if m:
                    actions.append(fmt(m))
                    break
        return actions

    # ---------- validacion ----------
    def _validate(self, stmt: str, domain: str, plan: list):
        """Devuelve (ok, motivo). Solo verifica; no resuelve."""
        if not plan:
            return False, "el plan esta vacio"
        world = _Mystery(stmt) if domain == "mystery" else _BlocksWorld(stmt)
        for i, a in enumerate(plan):
            p = a.strip("() ").split()
            err = world.apply(p[0], p[1:])
            if err:
                return False, f"paso {i + 1} '{a}' invalido: {err}"
        if not world.goal_reached():
            return False, "el plan no alcanza la meta"
        return True, "ok"

    # ---------- API ----------
    def solve(self, scenario_context: str, llm_engine_func) -> list:
        domain = self._domain(scenario_context)
        stmt = self._final_statement(scenario_context)
        base = self.build_prompt(scenario_context)

        raw = llm_engine_func(
            prompt=base, system=self.system_prompt, temperature=0.0, top_p=1.0,
            do_sample=False, enable_thinking=False, max_new_tokens=220)
        plan = self._parse_plan(raw, scenario_context)
        ok, motivo = self._validate(stmt, domain, plan)
        first = plan  # fallback: nunca entregamos peor que el intento base

        # Bucle de reparacion: si el plan es invalido, le devolvemos el error a
        # Qwen y REGENERA (Qwen sigue siendo quien produce el plan).
        for _ in range(self.max_repairs):
            if ok:
                return plan
            wrong = raw.split("[PLAN END]")[0].strip()
            repair = (f"{base}{wrong}\n[PLAN END]\n\n"
                      f"The plan above is INVALID: {motivo}. Rewrite the COMPLETE "
                      f"corrected plan that reaches the goal.\n[PLAN]")
            raw = llm_engine_func(
                prompt=repair, system=self.system_prompt, temperature=0.0, top_p=1.0,
                do_sample=False, enable_thinking=False, max_new_tokens=220)
            plan = self._parse_plan(raw, scenario_context)
            ok, motivo = self._validate(stmt, domain, plan)

        return plan if ok else first

    def complexity_level(self, plan: list) -> int:
        return len(plan)
