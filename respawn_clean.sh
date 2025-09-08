#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kill ROS nodes that publish/subscribe specific topics (supports regex).
Usage examples:
  ./kill_by_topic.py /gem/ackermann_cmd
  ./kill_by_topic.py --include-subscribers /gem/ackermann_cmd /cmd_vel
  ./kill_by_topic.py --regex '.*ackermann_cmd$' --keep '.*vision_lka.*'
"""

import argparse, re, time, subprocess, sys
import rosgraph

DEF_TIMEOUT = 5.0

def get_system_state():
    m = rosgraph.Master('/kill_by_topic')
    pubs, subs, srvs = m.getSystemState()
    # convert to dict: topic -> [nodes]
    pubs_d = {t: set(ns) for t, ns in pubs}
    subs_d = {t: set(ns) for t, ns in subs}
    return pubs_d, subs_d

def match_topics(all_topics, patterns, use_regex):
    hit = set()
    for t in all_topics:
        for p in patterns:
            if use_regex:
                if re.search(p, t):
                    hit.add(t)
            else:
                if t == p:
                    hit.add(t)
    return hit

def filter_nodes(nodes, keep_regexes):
    if not keep_regexes:
        return nodes
    out = set()
    for n in nodes:
        if any(re.search(kr, n) for kr in keep_regexes):
            continue
        out.add(n)
    return out

def kill_nodes(nodes):
    killed = []
    for n in sorted(nodes):
        try:
            subprocess.check_call(['rosnode', 'kill', n],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            killed.append(n)
        except subprocess.CalledProcessError:
            # node may already be dead
            pass
    return killed

def wait_gone(nodes, timeout=DEF_TIMEOUT):
    t0 = time.time()
    left = set(nodes)
    while time.time() - t0 < timeout and left:
        try:
            live = set(subprocess.check_output(['rosnode','list']).decode().split())
        except Exception:
            break
        left = {n for n in left if n in live}
        time.sleep(0.2)
    return left  # still alive

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('topics', nargs='+', help='topics to clean (exact or regex)')
    ap.add_argument('--regex', action='store_true', help='treat topics as regex patterns')
    ap.add_argument('--include-subscribers', action='store_true',
                   help='also kill subscribers (not only publishers)')
    ap.add_argument('--keep', action='append', default=[],
                   help='regex of node names to KEEP (do not kill), can repeat')
    ap.add_argument('--extra-kill', action='append', default=[],
                   help='regex of node names to force kill in addition (e.g. ".*cmd_relay_.*")')
    ap.add_argument('--timeout', type=float, default=DEF_TIMEOUT)
    args = ap.parse_args()

    pubs_d, subs_d = get_system_state()
    all_topics = set(pubs_d.keys()) | set(subs_d.keys())
    topics = match_topics(all_topics, args.topics, args.regex)

    if not topics:
        print("[kill_by_topic] No topics matched; nothing to do.", file=sys.stderr)
        return 0

    to_kill = set()
    for t in sorted(topics):
        pubs = pubs_d.get(t, set())
        subs = subs_d.get(t, set()) if args.include-subscribers else set()
        to_kill |= pubs | subs
        print(f"[kill_by_topic] topic: {t}")
        if pubs:
            print(f"  pubs: {', '.join(sorted(pubs))}")
        if subs and args.include-subscribers:
            print(f"  subs: {', '.join(sorted(subs))}")

    # Also kill any extra patterns
    try:
        live_nodes = set(subprocess.check_output(['rosnode','list']).decode().split())
    except Exception:
        live_nodes = set()
    for rx in args.extra_kill:
        rx_comp = re.compile(rx)
        to_kill |= {n for n in live_nodes if rx_comp.search(n)}

    # Filter keep
    victims = filter_nodes(to_kill, args.keep)
    if not victims:
        print("[kill_by_topic] Nothing

