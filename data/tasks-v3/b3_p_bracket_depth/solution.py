def max_bracket_depth(text):
    opens="([{"; closes=")]}"; match={")":"(","]":"[","}":"{"}
    stack=[]; best=0
    for ch in text:
        if ch in opens:
            stack.append(ch); best=max(best,len(stack))
        elif ch in closes:
            if not stack or stack.pop()!=match[ch]: return -1
    return -1 if stack else best
