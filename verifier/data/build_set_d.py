#!/usr/bin/env python3
"""Set D: 30 agentic tasks with objective graders. Each task is self-contained.

- debug_broken_repo  (10): TWO interacting bugs; hidden grader tests edge cases
- broken_env         (10): multi-part breakage recoverable via careful reading
- pipeline           (10): write non-trivial scripts end-to-end

Graders are objective: test_cmd exit code. Every task ships easy visible
tests but test_cmd runs a HIDDEN grader with edge cases the agent can't read.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "set_d_agentic.jsonl")


def T(id, category, instruction, files, test_cmd):
    return {"id": id, "category": category, "instruction": instruction,
            "files": files, "test_cmd": test_cmd}


TASKS = [
    # ---------- debug: two interacting bugs + hidden traps ----------
    T("D01", "debug",
      "calc.py: invoice_total() has TWO bugs: wrong rounding mode (banker's vs half-up) AND "
      "negative totals on extreme discounts. Fix both; hidden grader checks half-up rounding and floor at zero.",
      [
          ["calc.py",
           "def invoice_total(subtotal, discount_percent):\n"
           "    discount = subtotal * discount_percent / 100\n"
           "    total = subtotal - discount\n"
           "    return round(total)  # BUG 1: round() not half-up to 2dp; BUG 2: no floor at 0\n"],
          ["tests/test_calc.py",
           "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
           "from calc import invoice_total\nassert invoice_total(100,10)==90.0\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"from calc import invoice_total as f; import decimal; "
      "assert f(100,10)==90.0; assert f(100,150)==0.0; assert f(50,0)==50.0; "
      "assert f(33.33,10)==30.0; assert f(1.005*100,0)==100.5; print('HIDDEN PASS')\""),
    T("D02", "debug",
      "dedup.py: dedup() has TWO bugs — it drops the FIRST occurrence instead of keeping it, AND "
      "it mishandles unhashable/mixed types. Fix both. Hidden grader checks order and stability.",
      [
          ["dedup.py",
           "def dedup(items):\n"
           "    seen=set()\n"
           "    out=[]\n"
           "    for it in reversed(items):\n"
           "        if it not in seen:\n"
           "            seen.add(it); out.append(it)\n"
           "    return list(reversed(out))\n"],
          ["test_dedup.py",
           "from dedup import dedup\nassert dedup(['a','b','a'])==['a','b']\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"from dedup import dedup; "
      "assert dedup(['a','b','a','c','b'])==['a','b','c']; "
      "assert dedup([1,2,2,3,1])==[1,2,3]; "
      "assert dedup([(1,),(2,),(1,)])==[(1,),(2,)]; "
      "assert dedup(['A','a'])==['A','a']; print('HIDDEN PASS')\""),
    T("D03", "debug",
      "temperature.py: f_to_k() is wrong AND c_to_f() has a subtle precision bug. Fix both. "
      "Hidden grader checks the full chain with tight tolerance.",
      [
          ["temperature.py",
           "def c_to_f(c): return c*9/5+32\n"
           "def f_to_c(f): return (f-32)*5/9\n"
           "def c_to_k(c): return c+273.15\n"
           "def k_to_c(k): return k-273.15\n"
           "def f_to_k(f): return f+273.15  # BUG: F treated as C\n"],
          ["test_temperature.py",
           "import temperature as t\nassert abs(t.c_to_f(100)-212)<1e-6\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"import temperature as t; "
      "assert abs(t.f_to_k(32)-273.15)<1e-6; assert abs(t.f_to_k(212)-373.15)<1e-6; "
      "assert abs(t.k_to_c(t.c_to_k(25))-25)<1e-6; assert abs(t.f_to_c(t.c_to_f(-40))-(-40))<1e-6; print('HIDDEN PASS')\""),
    T("D04", "debug",
      "inventory.py: restock() has TWO bugs — it applies negative quantities (corrupting "
      "counts; negatives must be rejected as a no-op) AND it silently ignores restocking an "
      "item that is not yet tracked (new items must be created with the given quantity, "
      "same accumulation rule as add()). Fix both. Hidden grader checks both guards.",
      [
          ["inventory.py",
           "class Inventory:\n"
           "    def __init__(self): self.items={}\n"
           "    def add(self,n,q): self.items[n]=self.items.get(n,0)+q\n"
           "    def restock(self,n,q):\n"
           "        if n not in self.items: return  # BUG: silently ignores new items\n"
           "        self.items[n]=self.items.get(n,0)+q  # BUG: applies negative quantities\n"
           "    def total(self): return sum(self.items.values())\n"],
          ["test_inventory.py",
           "from inventory import Inventory\n"
           "i=Inventory(); i.add('a',5); i.restock('a',3); assert i.items['a']==8\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"from inventory import Inventory; "
      "i=Inventory(); i.add('a',5); i.restock('a',-2); assert i.items['a']==5; "
      "i.restock('zz',3); assert i.items.get('zz')==3; "
      "assert i.total()==8; print('HIDDEN PASS')\""),
    T("D05", "debug",
      "bank.py: transfer() has TWO bugs — it allows overdraft AND self-transfer double-counts. "
      "Fix both. Hidden grader checks negative-balance prevention and self-transfer no-op.",
      [
          ["bank.py",
           "class Bank:\n"
           "    def __init__(self): self.bal={}\n"
           "    def open(self,n,a=0): self.bal[n]=a\n"
           "    def transfer(self,s,d,amt):\n"
           "        self.bal[s]=self.bal.get(s,0)-amt\n"
           "        self.bal[d]=self.bal.get(d,0)+amt\n"
           "        return True\n"],
          ["test_bank.py",
           "from bank import Bank\n"
           "b=Bank(); b.open('a',100); b.open('b',0); assert b.transfer('a','b',40)\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"from bank import Bank; "
      "b=Bank(); b.open('a',100); b.open('b',0); "
      "assert not b.transfer('a','b',150); assert b.bal['a']==100 and b.bal['b']==0; "
      "assert not b.transfer('x','b',1); assert b.transfer('a','a',10); assert b.bal['a']==100; print('HIDDEN PASS')\""),
    T("D06", "debug",
      "parser.py: parse_records() has TWO bugs — it excludes any line containing a '#' "
      "character instead of only whole-line comments (a comment line is one whose first "
      "non-blank character is '#'; content after an inline '#' must be preserved verbatim) "
      "AND it strips leading/trailing whitespace from kept lines (whitespace must be "
      "preserved verbatim). Blank lines remain excluded. Fix both. Hidden grader checks "
      "inline comments and whitespace handling.",
      [
          ["parser.py",
           "def parse_records(text):\n"
           "    lines=text.splitlines()\n"
           "    out=[]\n"
           "    for l in lines:\n"
           "        if '#' in l: continue  # BUG: treats any '#' as a whole-line comment\n"
           "        out.append(l.strip())  # BUG: destroys surrounding whitespace\n"
           "    return [x for x in out if x]  # blank lines stay excluded (intended)\n"],
          ["test_parser.py",
           "from parser import parse_records\nassert parse_records('# h\\nalpha\\n')==['alpha']\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"from parser import parse_records; "
      "assert parse_records('a # inline\\nb')==['a # inline','b']; "
      "assert parse_records('#x\\n\\n y \\n')==[' y ']; print('HIDDEN PASS')\""),
    T("D07", "debug",
      "url_builder.py: build_url() has TWO bugs — it drops the scheme AND mishandles paths without "
      "a leading slash. Fix both. Hidden grader checks full URL correctness.",
      [
          ["url_builder.py",
           "def build_url(scheme,host,path):\n"
           "    return host+path  # BUG: drops scheme://\n"],
          ["test_url.py",
           "from url_builder import build_url\nassert build_url('https','example.com','/x')=='https://example.com/x'\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"from url_builder import build_url as f; "
      "assert f('https','example.com','x')=='https://example.com/x'; "
      "assert f('http','h','/')=='http://h/'; "
      "assert f('https','h','/a?b=1')=='https://h/a?b=1'; print('HIDDEN PASS')\""),
    T("D08", "debug",
      "collatz.py: next_collatz() has TWO bugs — even-branch uses //3 AND the base case returns 0 "
      "instead of 1. Fix both. Hidden grader walks the full sequence.",
      [
          ["collatz.py",
           "def next_collatz(n):\n"
           "    if n<=1: return 0  # BUG: should return 1\n"
           "    if n%2==0: return n//3  # BUG: should be n//2\n"
           "    return 3*n+1\n"],
          ["test_collatz.py",
           "from collatz import next_collatz\nassert next_collatz(10)==5\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"from collatz import next_collatz as f; "
      "n=27; c=0\nwhile n!=1: n=f(n); c+=1; assert c<200\nassert f(1)==1 and f(6)==3 and f(5)==16\nprint('HIDDEN PASS')\""),
    T("D09", "debug",
      "roman.py: to_roman() has TWO bugs — missing the 4/9 subtraction cases AND a wrong symbol "
      "for one value. Fix both. Hidden grader checks 4, 9, 49, 99, 999, 3999.",
      [
          ["roman.py",
           "def to_roman(n):\n"
           "    pairs=[(1000,'M'),(500,'D'),(100,'C'),(50,'L'),(10,'X'),(5,'V'),(1,'I')]\n"
           "    out=''\n"
           "    for v,s in pairs:\n"
           "        out+=s*(n//v); n%=v\n"
           "    return out\n"],
          ["test_roman.py",
           "from roman import to_roman\nassert to_roman(3)=='III'\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"from roman import to_roman as f; "
      "assert f(4)=='IV' and f(9)=='IX' and f(49)=='XLIX' and f(99)=='XCIX'; "
      "assert f(999)=='CMXCIX' and f(3999)=='MMMCMXCIX'; print('HIDDEN PASS')\""),
    T("D10", "debug",
      "sorting.py: stable_sort_by() has TWO bugs — reverse=True breaks stability AND it mutates "
      "the input list. Fix both. Hidden grader checks stability and non-mutation.",
      [
          ["sorting.py",
           "def stable_sort_by(items, keyfn, reverse=False):\n"
           "    items.sort(key=keyfn, reverse=reverse)  # BUG: mutates input\n"
           "    return items\n"],
          ["test_sorting.py",
           "from sorting import stable_sort_by\nassert stable_sort_by([(1,'a'),(0,'b')],lambda x:x[0])==[(0,'b'),(1,'a')]\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"from sorting import stable_sort_by as f; "
      "items=[('a',1),('b',0),('c',1),('d',0)]; orig=list(items); "
      "out=f(items,lambda x:x[1]); assert items==orig, 'mutated input'; "
      "assert out==[('b',0),('d',0),('a',1),('c',1)]; "
      "assert f(items,lambda x:x[1],True)==[('a',1),('c',1),('b',0),('d',0)]; print('HIDDEN PASS')\""),

    # ---------- broken environment ----------
    T("D11", "env",
      "pipeline.py imports 'libmath' AND references a function 'square'. Create package lib/ with "
      "lib/__init__.py exposing square(x)=x*x, patch pipeline.py, output RESULT 49.",
      [
          ["pipeline.py", "from libmath import square\n\nprint('RESULT', square(7))\n"],
      ],
      "python3 pipeline.py | grep -q 'RESULT 49'"),
    T("D12", "env",
      "config.ini has wrong DATA_DIR AND wrong delimiter (; instead of ,). inputs.csv uses commas. "
      "Fix config.ini so run_data.py prints OK.",
      [
          ["config.ini", "[paths]\nDATA_DIR = /nonexistent/share\ndelimiter = ;\n"],
          ["run_data.py",
           "import configparser, csv, os\n"
           "cfg=configparser.ConfigParser(); cfg.read('config.ini')\n"
           "path=os.path.join(cfg['paths']['DATA_DIR'],'inputs.csv')\n"
           "rows=list(csv.reader(open(path), delimiter=cfg['paths']['delimiter']))\n"
           "assert rows==[['a','b'],['1','2']]\nprint('OK')\n"],
          ["inputs.csv", "a,b\n1,2\n"],
      ],
      "python3 run_data.py | grep -q OK"),
    T("D13", "env",
      "app/__main__.py imports app.core but the dir is 'cor' AND app/__init__.py is empty (must stay "
      "importable). Reorganize so `python3 -m app` prints READY.",
      [
          ["app/__init__.py", ""],
          ["app/__main__.py", "from app.core import core\nprint('READY' if core.boot()==0 else 'BOOT FAIL')\n"],
          ["cor/__init__.py", "def boot():\n    return 0\n"],
      ],
      "python3 -m app | grep -q READY"),
    T("D14", "env",
      "loader.py uses yaml (missing). Rewrite with stdlib json loading config.json AND print LOADED "
      "only if stage=='prod' AND replicas>=1. No installs.",
      [
          ["loader.py",
           "import yaml\ncfg=yaml.safe_load(open('config.yaml'))\nassert cfg['stage']=='prod'\nprint('LOADED')\n"],
          ["config.json", "{\n  \"stage\": \"prod\",\n  \"replicas\": 3\n}\n"],
      ],
      "python3 loader.py | grep -q LOADED"),
    T("D15", "env",
      "toolchain.py crashes when gcc is missing AND also crashes on missing make. Make it print "
      "NOGCC/NOMAKE or GCC OK as appropriate, never crash.",
      [
          ["toolchain.py",
           "import subprocess\nr=subprocess.run(['gcc','--version'],capture_output=True)\nif r.returncode!=0: raise SystemExit('gcc missing')\nprint('GCC OK')\n"],
      ],
      "python3 toolchain.py | grep -qE 'GCC OK|NOGCC'"),
    T("D16", "env",
      "db_seed.py needs DB_URL env var. python-dotenv missing. Create .env AND patch db_seed.py to "
      "parse .env manually (not os.getenv on empty). Print DB READY.",
      [
          ["db_seed.py",
           "import os\nurl=os.environ.get('DB_URL')\nassert url and url.endswith('local.db')\nprint('DB READY')\n"],
      ],
      "python3 db_seed.py | grep -q 'DB READY'"),
    T("D17", "env",
      "casefile.py opens sample.txt (missing) AND expects 'hello-world' without trailing newline. "
      "Create the file exactly and make it print FILE OK.",
      [
          ["casefile.py",
           "path=os.environ.get('INPUT_FILE','sample.txt')\ntext=open(path).read()\nassert 'hello-world' in text\nprint('FILE OK')\n"],
      ],
      "echo -n 'hello-world' > sample.txt && python3 casefile.py | grep -q 'FILE OK'"),
    T("D18", "env",
      "install.sh runs pip install (network blocked) AND imports requests. Create vendor/requests.py "
      "stub (get()->None) and make install.sh succeed without network.",
      [
          ["install.sh", "#!/usr/bin/env bash\nset -e\npip install --target vendor -q requests && echo OK_INSTALL\n"],
      ],
      "bash install.sh >/dev/null 2>&1; python3 -c \"import sys,os; sys.path.insert(0,'vendor'); import requests; assert requests.get('x') is None; print('VENDOR OK')\" | grep -q 'VENDOR OK'"),
    T("D19", "env",
      "chmod_checker.py asserts util.sh is executable AND that it prints 'run' when executed. "
      "Make it executable (keep contents) and re-run.",
      [
          ["util.sh", "#!/bin/sh\necho run\n"],
          ["chmod_checker.py",
           "import os, subprocess\nassert os.access('util.sh', os.X_OK)\nout=subprocess.run(['./util.sh'],capture_output=True,text=True)\nassert out.stdout.strip()=='run'\nprint('EXEC OK')\n"],
      ],
      "python3 chmod_checker.py | grep -q 'EXEC OK'"),
    T("D20", "env",
      "legacy.cfg has count=-1 AND a stray 'mode = bad'. Change count to 7, remove/rename 'mode' to "
      "'mode = good', keep 'keep=2'. cfgcheck.py asserts all three.",
      [
          ["legacy.cfg", "[defaults]\ncount = -1\nkeep = 2\nmode = bad\n"],
          ["cfgcheck.py",
           "import configparser\nc=configparser.ConfigParser(); c.read('legacy.cfg')\n"
           "assert int(c['defaults']['count'])==7\nassert c['defaults']['keep']=='2'\n"
           "assert c['defaults']['mode']=='good'\nprint('CFG OK')\n"],
      ],
      "python3 cfgcheck.py | grep -q 'CFG OK'"),

    # ---------- multi-step pipeline ----------
    T("D21", "pipeline",
      "Write run.sh: create data/x.csv (header id,name; 2 rows), run csv2json.py -> out.json. "
      "Hidden grader checks exact out.json content.",
      [
          ["csv2json.py",
           "import csv,json\nrows=list(csv.DictReader(open('data/x.csv')))\njson.dump(rows,open('out.json','w'))\nprint('wrote',len(rows))\n"],
          ["tests/test_out.py",
           "import json\nrows=json.load(open('out.json'))\nassert len(rows)==2\nprint('VISIBLE PASS')\n"],
      ],
      "bash run.sh && python3 -c \"import json; r=json.load(open('out.json')); assert r==[{'id':'1','name':'a'},{'id':'2','name':'b'}]; print('HIDDEN PASS')\""),
    T("D22", "pipeline",
      "bundle.sh must cat data/*.txt in ALPHABETICAL order into bundle.txt. Create 3 files "
      "(a.txt,b.txt,c.txt). Hidden test checks exact order ['a1','b1','c1'].",
      [
          ["bundle.sh", "cat data/*.txt > bundle.txt\necho done\n"],
          ["tests/test_bundle.py",
           "lines=open('bundle.txt').read().split()\nassert len(lines)==3\nprint('VISIBLE PASS')\n"],
      ],
      "bash bundle.sh && python3 -c \"assert open('bundle.txt').read().split()==['a1','b1','c1'], open('bundle.txt').read(); print('HIDDEN PASS')\""),
    T("D23", "pipeline",
      "Write build.sh: numbers.py creates data/nums.json (1..5), stats.py writes stats.txt with "
      "sum=15. Hidden test checks stats.txt == 'sum=15' exactly.",
      [
          ["numbers.py",
           "import json,os\nos.makedirs('data',exist_ok=True)\njson.dump(list(range(1,6)),open('data/nums.json','w'))\nprint('ok')\n"],
          ["stats.py",
           "import json\nnums=json.load(open('data/nums.json'))\nopen('stats.txt','w').write(f'sum={sum(nums)}\\n')\nprint('ok')\n"],
          ["tests/test_stats.py", "assert open('stats.txt').read().strip()=='sum=15'\nprint('VISIBLE PASS')\n"],
      ],
      "bash build.sh && python3 -c \"assert open('stats.txt').read().strip()=='sum=15'; print('HIDDEN PASS')\""),
    T("D24", "pipeline",
      "Write merge.py: merge a.csv,b.csv,c.csv (key 'k') into merged.csv (k,v1,v2,v3). Hidden test "
      "checks column order and all rows.",
      [
          ["a.csv", "k,v1\nx,1\ny,2\n"],
          ["b.csv", "k,v2\nx,10\ny,20\n"],
          ["c.csv", "k,v3\nx,100\ny,200\n"],
          ["tests/test_merge_csv.py",
           "import csv\nrows=list(csv.DictReader(open('merged.csv')))\nassert len(rows)==2\nprint('VISIBLE PASS')\n"],
      ],
      "python3 merge.py && python3 -c \"import csv; rows=list(csv.DictReader(open('merged.csv'))); assert set(rows[0].keys())=={'k','v1','v2','v3'}; assert rows[0]=={'k':'x','v1':'1','v2':'10','v3':'100'}; print('HIDDEN PASS')\""),
    T("D25", "pipeline",
      "Write bench.py (no numpy) generating data/timing.txt with 3 ASCENDING lines epoch<i>=<ms>. "
      "Write bench.sh. Hidden test checks ascending.",
      [
          ["tests/test_bench.py",
           "lines=open('data/timing.txt').read().strip().splitlines()\nassert len(lines)==3\nprint('VISIBLE PASS')\n"],
      ],
      "bash bench.sh && python3 -c \"lines=open('data/timing.txt').read().strip().splitlines(); vals=[int(l.split('=')[1]) for l in lines]; assert vals==sorted(vals) and len(vals)==3; print('HIDDEN PASS')\""),
    T("D26", "pipeline",
      "Write sanitize.sh: messy.txt -> strip whitespace, drop blanks, write clean.txt. Hidden test "
      "checks clean.txt == ['alpha','beta','gamma'].",
      [
          ["messy.txt", "  alpha\n\n    beta    \n  gamma\n\n"],
          ["tests/test_clean.py",
           "lines=open('clean.txt').read().splitlines()\nassert len(lines)==3\nprint('VISIBLE PASS')\n"],
      ],
      "bash sanitize.sh && python3 -c \"assert open('clean.txt').read().splitlines()==['alpha','beta','gamma']; print('HIDDEN PASS')\""),
    T("D27", "pipeline",
      "cmdparse.py parses --key=value and --flag args into parsed.ini [args]. Currently hardcodes "
      "k=v. Rewrite the loop. Hidden test runs end-to-end.",
      [
          ["cmdparse.py",
           "import sys\nfrom configparser import ConfigParser\nc=ConfigParser(); c['args']={}\nfor a in sys.argv[1:]:\n    if a.startswith('--'): c['args']['k']='v'\nc.write(open('parsed.ini','w'))\nprint('PARSED')\n"],
          ["tests/test_cli.py",
           "import subprocess\nsubprocess.run(['python3','cmdparse.py','--stream=on'],check=True)\nprint('VISIBLE PASS')\n"],
      ],
      "python3 cmdparse.py --stream=on --verbose --count=7 && python3 -c \"import configparser; c=configparser.ConfigParser(); c.read('parsed.ini'); assert c['args']['stream']=='on'; assert c['args']['count']=='7'; assert 'verbose' in c['args']; print('HIDDEN PASS')\""),
    T("D28", "pipeline",
      "Write bump.sh: version.json (1.2.3) -> bump patch to 4, write back, write tag.txt 'v1.2.4'. "
      "Hidden test checks both files.",
      [
          ["version.json", "{\"major\":1,\"minor\":2,\"patch\":3}\n"],
          ["tests/test_version.py",
           "import json\nv=json.load(open('version.json'))\nassert v['patch']==4\nprint('VISIBLE PASS')\n"],
      ],
      "bash bump.sh && python3 -c \"import json; v=json.load(open('version.json')); assert v=={'major':1,'minor':2,'patch':4}; assert open('tag.txt').read().strip()=='v1.2.4'; print('HIDDEN PASS')\""),
    T("D29", "pipeline",
      "Write archive.sh: tar+gzip filedata/ into hello.tgz. Create filedata/hello.txt 'hi'. "
      "Hidden test checks hello.txt is in tarball.",
      [
          ["tests/test_archive.py",
           "import os\nassert os.path.exists('hello.tgz')\nprint('VISIBLE PASS')\n"],
      ],
      "bash archive.sh && python3 -c \"import tarfile; t=tarfile.open('hello.tgz'); assert any('hello.txt' in n for n in t.getnames()); print('HIDDEN PASS')\""),
    T("D30", "pipeline",
      "envaware.py reads PORT, defaults 8080, prints PORT=<port>. Hidden test runs with PORT cleared "
      "AND set. Fix it.",
      [
          ["envaware.py",
           "import os\nport=os.environ.get('PORT')\nprint('PORT='+(port or '8080'))\n"],
          ["tests/test_port.py",
           "import subprocess,os\nenv=dict(os.environ); env.pop('PORT',None)\nout=subprocess.run(['python3','envaware.py'],capture_output=True,text=True,env=env)\nassert out.stdout.strip()=='PORT=8080'\nprint('VISIBLE PASS')\n"],
      ],
      "python3 -c \"import subprocess,os; env=dict(os.environ); env.pop('PORT',None); o=subprocess.run(['python3','envaware.py'],capture_output=True,text=True,env=env); assert o.stdout.strip()=='PORT=8080'; env['PORT']='9090'; o=subprocess.run(['python3','envaware.py'],capture_output=True,text=True,env=env); assert o.stdout.strip()=='PORT=9090'; print('HIDDEN PASS')\""),
]

with open(OUT, "w") as f:
    for t in TASKS:
        f.write(json.dumps(t) + "\n")
print(f"wrote {OUT} ({len(TASKS)} tasks)")
