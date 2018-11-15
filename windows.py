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
                word = seq[i - j - 1]
                word = word if (word != '\n' or word != '\t') else ''

                temp_b += (word,)
            for j in range(n):
                word = seq[i + j + 1]
                word = word if (word != '\n' or word != '\t') else ''

                temp_c += (word,)

            total = temp_a + temp_b + (seq[i],) + temp_c
            windows.append(total)

        elif (i >= len(seq) - n):
            k = len(seq) - i - 1
            l = n - k
            temp_a = ()
            temp_b = ()
            temp_c = ()
            for j in reversed(range(n)):
                word = seq[i - j - 1]
                word = word if (word != '\n' or word != '\t') else ''
                temp_a += (word,)

            for j in range(k):
                word = seq[i + j + 1]
                word = word if (word != '\n' or word != '\t') else ''
                temp_b += (word,)

            for j in range(l):
                temp_c += ('',)

            word = seq[i] if (seq[i] != '\n' or seq[i] != '\t') else ''
            total = temp_a + (word,) + temp_b + temp_c
            windows.append(total)

        else:
            temp_a = ()
            temp_b = ()
            for j in reversed(range(n)):
                word = seq[i - j - 1]
                word = word if (word != '\n' or word != '\t') else ''
                temp_a += (word,)

            for j in range(n):
                word = seq[i + j + 1]
                word = word if (word != '\n' or word != '\t') else ''
                temp_b += (word,)

            word = seq[i] if (seq[i] != '\n' or seq[i] != '\t') else ''
            total = temp_a + (word,) + temp_b
            windows.append(total)

    return windows
