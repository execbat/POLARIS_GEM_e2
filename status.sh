#!/usr/bin/env bash
set -e
echo "=== NAV ==="
rostopic echo -n1 /gem/task_planner/state || true
echo -n "pure_pursuit status: "; rostopic echo -n1 /pure_pursuit/status || true
echo "cmd(ctrl):"; rostopic echo -n1 /pure_pursuit/cmd || true
echo "cmd(mux->wheels):"; rostopic echo -n1 /gem/ackermann_cmd || true
echo "Publishers /gem/ackermann_cmd:"; rostopic info /gem/ackermann_cmd | sed -n '/Publishers:/,/Subscribers:/p'
echo "=== SAFETY ==="
echo -n "e-stop: "; rostopic echo -n1 /gem/safety/stop || true
echo "=== ODOM/TIME ==="
rostopic echo -n1 /gem/odom || true
echo -n "/clock: "; rostopic echo -n1 /clock || true
echo "=== PARAMS (planner) ==="
rosparam get /gem/task_planner 2>/dev/null | sed -n '1,200p' || true
echo "=== PARAMS (pure_pursuit) ==="
rosparam get /gem/pure_pursuit 2>/dev/null | sed -n '1,200p' || true
echo "=== ROS_CONTROL ==="
rosservice call /gem/controller_manager/list_controllers "{}" || true
