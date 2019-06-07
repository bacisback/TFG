with concurrent.futures.ThreadPoolExecutor(max_workers= min(len(self.tests),ncores)) as executor:
			futures = {executor.submit(test.generate_histogram,sample_size) for test in self.tests}
			concurrent.futures.wait(futures)
