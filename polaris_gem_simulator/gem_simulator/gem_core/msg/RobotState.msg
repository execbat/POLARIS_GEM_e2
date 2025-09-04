int8 IDLE=0
int8 RUNNING=1
int8 ERROR=2

int8 state           # current state
uint32 error_mask    # bitmask of active errors
string reason        # error description
