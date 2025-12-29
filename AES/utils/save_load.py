import os
import pickle
import hickle

def save_hickle_file(filename, data):
    filename = filename + '.hickle'
    print ('Saving to %s' % filename)

    with open(filename, 'w') as f:
        #hickle.dump(data, f, mode='w')
        hickle.dump(data, filename, mode='w')

def load_hickle_file(filename):
    filename = filename + '.hickle'
    if os.path.isfile(filename):
        print ('Loading %s ...' % filename)
        data = hickle.load(filename)
        return data
    return None

def save_pickle_file(filename, data):
    filename = filename + '.pickle'
    print ('Saving to %s' % filename,)

    with open(filename, 'wb') as f:
        try:
            pickle.dump(data, f)
        except Exception:
            print ('Cannot pickle to %s' % filename)

def load_pickle_file(filename):
    filename = filename + '.pickle'
    if os.path.isfile(filename):
        print ('Loading %s ...' % filename,)
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return data
    return None


def write_out(y_test_p, filename):
    lines = []
    lines.append('preictal')
    for i in range(len(y_test_p)):
        lines.append('%.4f' % ((y_test_p[i])))
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
    print ('wrote', filename)