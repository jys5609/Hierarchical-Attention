# -*- coding: utf-8 -*-

"""
This script shows the summarized results based the CSV file generated by score_main.py
"""

import sys, argparse

SCHEDULES = [1,2]

def main(argv):
    #
    # CMD LINE ARGS
    # 
    parser = argparse.ArgumentParser(description='Formats a scorefile into a report and prints to stdout.')
    parser.add_argument('--scorefile',dest='csv',action='store',required=True,metavar='CSV_FILE',
                        help='File to read with CSV scoring data')
    args = parser.parse_args()

    #  "topic, slot, schedule, stat, N, result"
    header = True
    tables = {}

    tables['all'] = {}
    for schedule in SCHEDULES:
        tables['all'][schedule] = {}
        
    basic_stats = {}
    
    for line in open(args.csv):
        if header:
            header = False
            continue
        topic, slot, schedule, stat, N, result = [item.strip() for item in line.split(",")]
        
        stat = stat.strip()
        if topic == "basic" :
            basic_stats[slot] = result.strip()
        else :
            N = int(N)
            schedule = int(schedule)
            result = result.strip()
            if result != "-" :
                result = "%.7f" % float(result)
            
            if topic == 'all' and slot == 'all':
                tables['all'][schedule][stat] = result

    print
    print '                       featured metrics'
    print_row(["","all.schedule1","all.schedule2"],header=True)
    print_row(["segment.accuracy",tables["all"][1]["acc"],tables["all"][2]["acc"]])
    print_row(["slot_value.precision",tables["all"][1]["precision"],tables["all"][2]["precision"]])
    print_row(["slot_value.recall",tables["all"][1]["recall"],tables["all"][2]["recall"]])
    print_row(["slot_value.fscore",tables["all"][1]["f1"],tables["all"][2]["f1"]])
    print "\n\n"
    
    
    print '                                    basic stats'
    print '-----------------------------------------------------------------------------------'
    for k in sorted(basic_stats.keys()):
        v = basic_stats[k]
        print '%25s : %s' % (k,v)

    path = args.csv.split('/')[0]
    file = args.csv.split('/')[1]

    print "PATH NAME: ", path
    print "FILE NAME: ", file

    f = open('results/' + file + '.txt', 'w')

    f.write('\n')

    f.write(file)

    f.write('\n')
    f.write('\n')

    for i in return_row(["", "all.schedule1", "all.schedule2"], header=True):
        f.write(i)
        f.write('\n')

    f.write(return_row(["segment.accuracy", tables["all"][1]["acc"], tables["all"][2]["acc"]])[0])
    f.write('\n')
    f.write(return_row(["slot_value.precision", tables["all"][1]["precision"], tables["all"][2]["precision"]])[0])
    f.write('\n')
    f.write(return_row(["slot_value.recall", tables["all"][1]["recall"], tables["all"][2]["recall"]])[0])
    f.write('\n')
    f.write(return_row(["slot_value.fscore", tables["all"][1]["f1"], tables["all"][2]["f1"]])[0])
    f.write('\n')

    f.close()


def return_row(row, header=False):
    out = [str(x) for x in row]
    for i in range(len(out)):
        if i == 0:
            out[i] = out[i].ljust(20)
        else:
            out[i] = out[i].center(20)

    out = ("|".join(out))[:-1] + "|"

    result = []

    if header:
        result.append("-" * len(out))
        result.append(out)
        result.append("-" * len(out))

    else:
        result.append(out)

    return result

def print_row(row, header=False):
    out = [str(x) for x in row]
    for i in range(len(out)):
        if i==0 :
            out[i] = out[i].ljust(20)
        else :
            out[i] = out[i].center(20)
    
    out = ("|".join(out))[:-1]+"|"
    
    if header:
        print "-"*len(out)
        print out
        print "-"*len(out)
    else:
        print out


if (__name__ == '__main__'):
    main(sys.argv)
