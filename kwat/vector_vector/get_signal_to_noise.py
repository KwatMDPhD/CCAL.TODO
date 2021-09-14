from numpy import absolute


def get_signal_to_noise(ve1, ve2):

    me1 = ve1.mean()

    me2 = ve2.mean()

    st1 = ve1.std()

    st2 = ve2.std()

    fa = 0.2

    lo1 = absolute(me1) * fa

    lo2 = absolute(me2) * fa

    if me1 == 0:

        me1 = 1

        st1 = fa

    elif st1 < lo1:

        st1 = lo1

    if me2 == 0:

        me2 = 1

        st2 = fa

    elif st2 < lo2:

        st2 = lo2

    return (me2 - me1) / (st1 + st2)
