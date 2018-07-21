#Seq - the sequence of words of a given text
#n - the size of the window
def get_windows(seq, n):
    windows = []
    for i in range(len(seq)):
        if (i - n < 0):
            k = abs(i - n)
            l = abs(k - n)
            temp_a = ()
            temp_b = ()
            temp_c = ()
            for j in range(k):
                temp_a += ('',)
            for j in reversed(range(l)):
                temp_b += (seq[i - j - 1],)
            for j in range(n):
                temp_c += (seq[i + j + 1],)
            total = temp_a + temp_b + (seq[i],) + temp_c
            windows.append(total)
        elif (i >= len(seq) - n):
            k = len(seq) - i - 1
            l = n - k
            temp_a = ()
            temp_b = ()
            temp_c = ()
            for j in reversed(range(n)):
                temp_a += (seq[i - j - 1],)
            for j in range(k):
                temp_b += (seq[i + j + 1],)
            for j in range(l):
                temp_c += ('',)
            total = temp_a + (seq[i],) + temp_b + temp_c
            windows.append(total)
        else:
            temp_a = ()
            temp_b = ()
            for j in reversed(range(n)):
                temp_a += (seq[i - j - 1],)
            for j in range(n):
                temp_b += (seq[i + j + 1],)
            total = temp_a + (seq[i],) + temp_b
            windows.append(total)

    return windows
