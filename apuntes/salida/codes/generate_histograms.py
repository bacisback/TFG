with concurrent.futures.ThreadPoolExecutor(max_workers= len(self.tests)) as executor:
			futures = {executor.submit(test.generate_histogram,sample_size) for test in self.tests}
			concurrent.futures.wait(futures)