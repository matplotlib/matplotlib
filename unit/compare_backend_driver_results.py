from __future__ import print_function
import sys


def parse_results(filename):
    results = {}
    fd = open(filename, 'r')
    section = "???"
    for line in fd.readlines():
        line = line.strip()
        if line.startswith("testing"):
            section = line.split(" ", 1)[1]
            results.setdefault(section, {})
        elif line.startswith("driving"):
            driving, test, time = [x.strip() for x in line.split()]
            time = float(time)
            results[section][test] = time
    fd.close()
    return results


def check_results_are_compatible(results_a, results_b):
    for section in results_a.keys():
        if not section in results_b:
            raise RuntimeError("Backend '%s' in first set, but not in second" % section)

    for section in results_b.keys():
        if not section in results_a:
            raise RuntimeError("Backend '%s' in second set, but not in first" % section)


def compare_results(results_a, results_b):
    check_results_are_compatible(results_a, results_b)

    sections = results_a.keys()
    sections.sort()
    for section in results_a.keys():
        print("backend %s" % section)
        print("    %-40s %6s %6s %6s %6s" % ("test", "a", "b", "delta", "% diff"))
        print("    " + '-' * 69)
        deltas = []
        section_a = results_a[section]
        section_b = results_b[section]
        for test in section_a.keys():
            if test not in section_b:
                deltas.append([None, None, section_a[test], None, test])
            else:
                time_a = section_a[test]
                time_b = section_b[test]
                deltas.append([time_b / time_a, time_b - time_a, time_a, time_b, test])
        for test in section_b.keys():
            if test not in section_a:
                deltas.append([None, None, None, section_b[test], test])

        deltas.sort()
        for diff, delta, time_a, time_b, test in deltas:
            if diff is None:
                if time_a is None:
                    print("    %-40s    ??? % 6.3f    ???    ???" % (test, time_b))
                else:
                    print("    %-40s % 6.3f    ???    ???    ???" % (test, time_a))
            else:
                print("    %-40s % 6.3f % 6.3f % 6.3f %6d%%" % (test, time_a, time_b, delta, diff * 100))


if __name__ == '__main__':
    results_a_filename = sys.argv[-2]
    results_b_filename = sys.argv[-1]

    results_a = parse_results(results_a_filename)
    results_b = parse_results(results_b_filename)

    compare_results(results_a, results_b)
