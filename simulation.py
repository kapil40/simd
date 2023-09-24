import sys
import logging
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

from system import *
from statistics import *

class Simulation:

    logger = logging.getLogger("sim")

    def __init__(self, mission_time, iterations, raid_type, raid_num, disk_capacity, 
            disk_fail_parms, disk_repair_parms, disk_lse_parms, disk_scrubbing_parms, 
            disk_throughput, repair_bandwidth_percentage, disk_size, array_size,  
            force_re, required_re, fs_trace, filelevel, dedup, weighted, output_file):
        self.mission_time = mission_time
        self.iterations = iterations
        self.raid_type = raid_type
        self.raid_num = raid_num
        self.disk_capacity = disk_capacity
        self.disk_fail_parms = disk_fail_parms
        self.disk_repair_parms = disk_repair_parms
        self.disk_lse_parms = disk_lse_parms
        self.disk_scrubbing_parms = disk_scrubbing_parms
        self.disk_throughput = disk_throughput
        self.repair_bandwidth_percentage = repair_bandwidth_percentage
        self.disk_size = disk_size
        self.array_size = array_size
        self.fail_count = 0
        self.logger.debug("Simulation: iterations = %d" % iterations)

        self.system = None

        # The required relative error
        self.force_re = force_re
        self.required_re = required_re

        self.raid_failure_samples = Samples()
        self.lse_samples = Samples()

        self.systems_with_raid_failures = 0
        self.systems_with_lse = 0
        self.systems_with_data_loss = 0
        self.samples=[]

        self.cur_i = 0
        self.more_iterations = 0

        self.fs_trace = fs_trace
        self.filelevel = filelevel 
        self.dedup = dedup
        self.weighted = weighted
        
        self.start_time = datetime.datetime.now()

        self.output = None
        if output_file is not None:
            self.output = open(output_file, "a")

    def get_runtime(self):
        delta = datetime.datetime.now() - self.start_time
        d = delta.days
        s = delta.seconds%60
        m = (delta.seconds/60)%60
        h = delta.seconds/60/60
        return (d, h, m, s)

    def run_iterations(self, iterations):
        
        c=0
        for self.cur_i in range(iterations):
            # print(self.cur_i)
            if (self.cur_i & 16383) == 0:
                progress = 1.0*self.cur_i/self.iterations
                num = int(progress * 100)
                print("%6.2f%%: [" % (progress*100), "\b= "*num, "\b\b>", " "*(100-num), "\b\b]", "%3dd%2dh%2dm%2ds \r" % self.get_runtime(), file=sys.stderr, end='')


            self.system.reset()
            # c=c+1
            # print(c)
            result, count, event_type, samples = self.system.run()
            # self.queue.append(event_type)
            self.samples = samples
               
            self.fail_count = count
            if result[0] != 0 or result[1] != 0:
                self.systems_with_data_loss += 1

                if result[0] != 0:
                    self.logger.debug("%dth iteration: %s, %d bytes lost" % (self.cur_i, "RAID Failure", result[0]))
                    self.systems_with_raid_failures += 1
                    if(self.systems_with_raid_failures == 1):
                        print("First failure at iteration: " + str(self.cur_i))
                    #print "%f" % result[0]

                    if self.output:
                        print >>self.output, "R=%.6f" % result[0],

                if result[1] != 0:
                    self.logger.debug("%dth iterations: %s, %d bytes lost" % (self.cur_i, "Sectors Lost", result[1]))
                    self.systems_with_lse += 1
                    if self.output:
                        print >>self.output, "S=%d" % result[1],

                if self.output is not None:
                    print >>self.output, ""

            self.raid_failure_samples.addSample(result[0])
            self.lse_samples.addSample(result[1])
        
        print("%6.2f%%: [" % (100.0), "\b= "*100, "\b\b>", "\b]", "%3dd%2dh%2dm%2ds" % self.get_runtime(), file=sys.stderr) 
        mean = np.mean(samples)
        median = np.median(samples)

        if len(samples) == 0 or np.isnan(samples).any():
            print("Samples array is empty or contains NaN. Cannot calculate mean and median.")
        # Print mean and median
        else:
            print("Mean:", mean)
            print("Median:", median)
#         plt.hist(samples, bins=30, density=True, alpha=0.6, color='b', label='Histogram')

# # Plot the PDF of the Weibull distribution
#         x = np.linspace(0, np.max(samples), 1000)
#         pdf = weibull_min.pdf(x, shape, loc=location, scale=scale)
#         plt.plot(x, pdf, 'r-', lw=2, label='PDF')

#         plt.xlabel('x')
#         plt.ylabel('Probability Density')
#         plt.title('Weibull Distribution')
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        # print(self.samples)
        # print(len(self.samples))
    def simulate(self):

        self.system = System(self.mission_time, self.raid_type, self.raid_num, self.disk_capacity, self.disk_fail_parms,
            self.disk_repair_parms, self.disk_lse_parms, self.disk_scrubbing_parms,
            self.disk_throughput, self.repair_bandwidth_percentage, self.disk_size, self.array_size, 
            self.fs_trace, self.filelevel, self.dedup, self.weighted)

        self.more_iterations = self.iterations
        start_time = time.time()
        while True:

            self.run_iterations(self.more_iterations)

            self.raid_failure_samples.calcResults("0.95")
            self.lse_samples.calcResults("0.95")

            if self.force_re == False:
                break

            if self.raid_failure_samples.value_re > self.lse_samples.value_re and self.raid_failure_samples.value_re > self.required_re:
                self.more_iterations = int((self.raid_failure_samples.value_re/self.required_re - 1) * self.iterations)
                print("Since RAID FAILURE Relative Error %5f > %5f," % (self.raid_failure_samples.value_re, self.required_re), file=sys.stderr)
                # print >> sys.stderr, "Since RAID FAILURE Relative Error %5f > %5f," % (self.raid_failure_samples.value_re, self.required_re),
            elif self.lse_samples.value_re > self.required_re:
                self.more_iterations = int((self.lse_samples.value_re/self.required_re - 1) * self.iterations)
                print("Since LSE Relative Error %5f > %5f," % (self.lse_samples.value_re, self.required_re), file=sys.stderr)
                # print >> sys.stderr, "Since LSE Relative Error %5f > %5f," % (self.lse_samples.value_re, self.required_re),
            else:
                break

            if self.more_iterations < 10000:
                self.more_iterations = 10000

            print("%d more iterations to meet the requirement." % self.more_iterations, file=sys.stderr)

            self.iterations += self.more_iterations

        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        if self.output is not None:
            print >>self.output, "I=%d"%self.iterations
            self.output.close()

        # finished, return results
        # the format of result:
        # print("Total Disks failures:", self.fail_count)
        return (self.system.dedup_model, self.raid_failure_samples, self.lse_samples, self.systems_with_data_loss, self.systems_with_raid_failures, 
                self.systems_with_lse, self.iterations, self.system.get_df(),self.samples)
