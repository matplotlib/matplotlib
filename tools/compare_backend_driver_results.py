import sys


def parse_results(filename):
    results = {}
    section = "???"
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("testing"):
                section = line.split(" ", 1)[1]
                results.setdefault(section, {})
            elif line.startswith("driving"):
                driving, test, time = [x.strip() for x in line.split()]
                time = float(time)
                results[section][test] = time
    return results


def check_results_are_compatible(results_a, results_b):
    a_minus_b = {*results_a} - {*results_b}
    if a_minus_b:
        raise RuntimeError(
            "Backends {} in first set, but not in second".format(a_minus_b))
    b_minus_a = {*results_b} - {*results_a}
    if b_minus_a:
        raise RuntimeError(
            "Backends {} in second set, but not in first".format(b_minus_a))


def compare_results(results_a, results_b):
    check_results_are_compatible(results_a, results_b)

    sections = results_a.keys()
    sections.sort()
    for section in results_a.keys():
        print(f"backend {section}" % section)
        print(f"    {'test':40} {'a':>6} {'b':>6} {'delta':>6} {'% diff':>6}")
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
                deltas.append(
                    [time_b / time_a, time_b - time_a, time_a, time_b, test])
        for test in section_b.keys():
            if test not in section_a:
                deltas.append([None, None, None, section_b[test], test])

        deltas.sort()
        for diff, delta, time_a, time_b, test in deltas:
            if diff is None:
                if time_a is None:
                    print(f"    {test:40}    ??? {time_b: 6.3f}    ???    ???")
                else:
                    print(f"    {test:40} {time_a: 6.3f}    ???    ???    ???")
            else:
                print(f"    {test:40} {time_a: 6.3f} {time_b: 6.3f} "
                      f"{delta: 6.3f} {diff:6.0%}")


if __name__ == '__main__':
    results_a_filename = sys.argv[-2]
    results_b_filename = sys.argv[-1]

    results_a = parse_results(results_a_filename)
    results_b = parse_results(results_b_filename)

    compare_results(results_a, results_b)
