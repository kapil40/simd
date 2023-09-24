import sys
import logging
import itertools
from collections import deque
import random
from component import *
# use array instead of list to reduce memory overhead
import array

class DeduplicationModel:

	def __init__(self, trace, filelevel, dedup, weighted):
		self.trace = trace
		self.filelevel = filelevel 
		self.dedup = dedup
		self.weighted = weighted
		self.df = 1.0

	def raid_failure(self, corrupted_area):
		return None

	def sector_error(self, lse_count):
		return None

class DeduplicationModel_Chunk_NoDedup(DeduplicationModel):
	def __init__(self, weighted):
		self.filelevel = False
		self.dedup = False

		self.weighted = weighted
		self.df = 1.0

	# percent of corrupted blocks
	def raid_failure(self, corrupted_area):
		return corrupted_area

	# the size of corrupted 8KB blocks 
	def sector_error(self, lse_count):
		# multiply with file system block
		if self.weighted:
			return lse_count * 8192
		else:
			return lse_count

# reference count / chunk size * reference count
# reference count / chunk size * reference count
# ...
# D/F
# 0% progress
# 1% progress
# ...
# 100% progress
class DeduplicationModel_Chunk_Dedup(DeduplicationModel):
	def __init__(self, trace, weighted):
		self.filelevel = False
		self.dedup = True

		self.df = 0
		# array of uncorrectable sector errors
		self.use_array = array.array("l")
		# array of raid failure
		self.rf_array = array.array("f")

		self.weighted = weighted 
		self.trace = trace
		tracefile = open(self.trace, "r")
		if self.weighted:
			assert(tracefile.readline() == "CHUNK:DEDUP:WEIGHTED\n")
		else:
			assert(tracefile.readline() == "CHUNK:DEDUP:NOT WEIGHTED\n")

		for line in tracefile:
			if line[:-1].isdigit() == True:
				self.use_array.append(int(line))
			else:
				assert(self.df == 0)
				self.df = float(line)
				break;
		for line in tracefile:
			self.rf_array.append(float(line))
		assert(self.df >= 1)
		assert(len(self.rf_array) == 101)

		tracefile.close()

	# percent of corrupted logical chunks
	def raid_failure(self, corrupted_area):
		index = int((corrupted_area+0.005)*100)
		assert(index >= 0 and index <= 100)
		result = 1.0 - self.rf_array[-1-index]
		return result

	# the size of corrupted logical chunks
	def sector_error(self, lse_count):
		lost = 0;
		for i in xrange(lse_count):
			lost += self.use_array[random.randrange(len(self.use_array))]

		assert(lost>=0)
		return lost 

# 0% progress
# 1% progress
# ...
# 100% progress
class DeduplicationModel_File_NoDedup_NotWeighted(DeduplicationModel):
	def __init__(self, trace):
		self.filelevel = True
		self.dedup = False
		self.weighted = False

		self.trace = trace
		tracefile = open(self.trace, "r")

		assert(list(itertools.islice(tracefile, 1))[0] == "FILE:NO DEDUP:NOT WEIGHTED\n")

		# Totally 101 items for RAID failures
		self.rf_array = [float(i) for i in itertools.islice(tracefile, 0, None)]
		self.df = 1.0

	# percent of corrupted files
	def raid_failure(self, corrupted_area):
		index = int((corrupted_area+0.005)*100)
		assert(index >= 0 and index <= 100)
		result = 1.0 - self.rf_array[-1-index]
		return result

	# number of corrupted files
	def sector_error(self, lse_count):
		return lse_count 

# file size for chunk 1 
# file size for chunk 2 
# ...
# 0% progress
# 1% progress
# ...
# 100% progress
class DeduplicationModel_File_NoDedup_Weighted(DeduplicationModel):
	def __init__(self, trace):
		self.filelevel = True
		self.dedup = False
		self.weighted = True

		self.df = 0
		self.trace = trace
		tracefile = open(self.trace, "r")

		# array of uncorrectable sector errors
		self.use_array = array.array("l")
		# array of raid failure
		self.rf_array = array.array("f")

		assert(tracefile.readline() == "FILE:NO DEDUP:WEIGHTED\n")

		for line in tracefile:
			if int(line) == 0:
				self.df = 1.0
				self.rf_array.append(float(line))
				break
			else:
				self.use_array.append(int(line))
		for line in tracefile:
			self.rf_array.append(float(line))

		assert(self.df >= 1)
		assert(len(self.rf_array) == 101)

		tracefile.close()

	# percent of corrupted files in size
	def raid_failure(self, corrupted_area):
		index = int((corrupted_area+0.005)*100)
		assert(index >= 0 and index <= 100)
		result = 1.0 - self.rf_array[-1-index]
		return result 

	# size of corrupted files
	def sector_error(self, lse_count):
		bytes_lost = 0;
		for i in xrange(lse_count):
			bytes_lost += self.use_array[random.randrange(len(self.use_array))]

		assert(bytes_lost>=0)
		return bytes_lost

# referred file size (MODE C)/count (MODE B) for chunk 1 
# referred file size/count for chunk 2 
# ...
# D/F
# 0% progress
# 1% progress
# ...
# 100% progress
class DeduplicationModel_File_Dedup(DeduplicationModel):
	def __init__(self, trace, weighted):
		self.filelevel = True
		self.dedup = True
		self.weighted = weighted
		self.trace = trace
		tracefile = open(self.trace, "r")

		if self.weighted:
			assert(tracefile.readline() == "FILE:DEDUP:WEIGHTED\n")
		else:
			assert(tracefile.readline() == "FILE:DEDUP:NOT WEIGHTED\n")

		self.df = 0

		# array of uncorrectable sector errors
		self.use_array = array.array("l")
		# array of raid failure
		self.rf_array = array.array("f")

		# The last 101 items are for RAID failures
		for line in tracefile:
			if line[:-1].isdigit() == True:
				self.use_array.append(int(line))
			else:
				assert(self.df == 0)
				self.df = float(line)
				break;
		for line in tracefile:
			self.rf_array.append(float(line))

		assert(self.df >= 1)
		assert(len(self.rf_array) == 101)

		tracefile.close()

	# percent of corrupted files in number or size
	def raid_failure(self, corrupted_area):
		index = int((corrupted_area+0.005)*100)
		assert(index >= 0 and index <= 100)
		result = 1.0 - self.rf_array[-1-index]
		return result

	# number or size of corrupted files
	def sector_error(self, lse_count):
		corrupted_files = 0
		for i in xrange(lse_count):
			corrupted_files += self.use_array[random.randrange(len(self.use_array))]

		assert(corrupted_files>=0)
		return corrupted_files

class System:
	#RESULT_NOTHING_LOST = 0 #"Nothing Lost"
	#RESULT_RAID_FAILURE = 1 #"RAID Failure"
	#RESULT_SECTORS_LOST = 2 #"Sectors Lost"
	
	logger = logging.getLogger("sim")
	

	# A system consists of many RAIDs
	def __init__(self, mission_time, raid_type, raid_num, disk_capacity, 
			disk_fail_parms, disk_repair_parms, disk_lse_parms, disk_scrubbing_parms,
			disk_throughput, repair_bandwidth_percentage, disk_size, array_size, 
			trace, filelevel, dedup, weighted):
		self.mission_time = mission_time
		self.raid_num = raid_num
		self.avail_raids = raid_num
		self.fail_count = 0
		self.samples = []
		self.logger.debug("System: mission_time = %d, raid_num = %d" % (self.mission_time, self.raid_num))
		self.event_type_first = ""
		self.event_queue = None

		self.raids = [Raid(raid_type, disk_capacity, disk_fail_parms,
			disk_repair_parms, disk_lse_parms, disk_scrubbing_parms,
			disk_throughput, repair_bandwidth_percentage, disk_size, array_size) for i in range(raid_num)]

		if filelevel == False:
			if dedup == False:
				self.dedup_model = DeduplicationModel_Chunk_NoDedup(weighted)
			else:
				self.dedup_model = DeduplicationModel_Chunk_Dedup(trace, weighted)
		else:
			if dedup == False and weighted == False:
				self.dedup_model = DeduplicationModel_File_NoDedup_NotWeighted(trace);
			elif dedup == False and weighted == True:
				self.dedup_model = DeduplicationModel_File_NoDedup_Weighted(trace);
			elif dedup == True:
				self.dedup_model = DeduplicationModel_File_Dedup(trace, weighted)

	def reset(self):
		self.event_queue = []
		for r_idx in range(len(self.raids)):
			self.event_queue.extend(self.raids[r_idx].reset(r_idx, self.mission_time))

		self.event_queue = sorted(self.event_queue, reverse=True)

	def calc_bytes_lost(self):
		results = [0, 0]
		for raid in self.raids:
			if(raid.state == Raid.RAID_STATE_FAILED):
				# Not support multiple RAIDs in this model
				assert(raid.corrupted_area >= 0 and raid.corrupted_area <= 1)
				results[0] = self.dedup_model.raid_failure(raid.corrupted_area)

			results[1] += self.dedup_model.sector_error(raid.lse_count)

		return results

	def go_to_next_event(self):
		
		while True:
			if len(self.event_queue) == 0:
				return None
			(event_time, disk_idx, raid_idx) = self.event_queue.pop()
			# print(self.event_queue)
			if self.raids[raid_idx].state == Raid.RAID_STATE_OK:
				break

		# After update, the system state is consistent
		(event_type, next_event_time, rebuild_time,res,samples) = self.raids[raid_idx].update_to_event(event_time, disk_idx)
		self.samples = samples
		self.fail_count = res
		# print(next_event_time)
		# if event_type == "event disk fail":
		# 	print(event_type, next_event_time, rebuild_time)
		if next_event_time <= self.mission_time:
			self.event_queue.append((next_event_time, disk_idx, raid_idx))
			size = len(self.event_queue)
			if size >= 2 and self.event_queue[size - 1] > self.event_queue[size - 2]:
				self.event_queue = sorted(self.event_queue, reverse=True)				

		self.logger.debug("raid_idx = %d, disk_idx = %d, event_type = %s, event_time = %d" % (raid_idx, disk_idx, event_type, event_time))
		return (event_type, event_time, raid_idx)

	# Three possible returns
	# [System.RESULT_NOTHING_LOST, 0]
	# [System.RESULT_RAID_FAILURE, bytes]
	# [System.RESULT_SECTORS_LOST, bytes]
	def run(self):
		
		while True:

			e = self.go_to_next_event()
			
			if e == None:
				break

			(event_type, event_time, raid_idx) = e
			# print(event_type)
			self.event_type_first = event_type
			if event_type == Disk.DISK_EVENT_REPAIR:
				continue
			
			# Check whether the failed disk causes a RAID failure 
			if self.raids[raid_idx].check_failure(event_time) == True:
				# TO-DO: When deduplication model is ready, we need to amplify bytes_lost
				# e.g., bytes_lost * deduplication factor
				# print("yes")
				self.avail_raids -= 1
				# print("test")
				if self.avail_raids == 0:
					break

				continue
				
			# Check whether a LSE will cause a data loss
			# check_sectors_lost will return the bytes of sector lost
			# We need to further amplify it according to our file system
			self.raids[raid_idx].check_sectors_lost(event_time)
		# print(self.fail_count)
		# The mission concludes or all RAIDs fail
		return (self.calc_bytes_lost(),self.fail_count,self.event_type_first,self.samples)

	def get_df(self):
		return self.dedup_model.df
