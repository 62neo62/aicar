from threading import Timer
import back_wheels
import logging
import time

bw = back_wheels.Back_Wheels(debug=False, db=db_file) 

global SPEED
SPEED = 60 #max duty cycle

class TrafficObject(object):

    def set_car_state(self, car_state):
        pass


class redlight(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('redlight: stopping car')
        bw.speed = 0 # 0% duty cycle
        wait_done(3)
        bw.speed = SPEED
        
class greenlight(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('greenlight: make no changes')
        bw.speed = SPEED

class pedestrian(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('pedestrian: stopping car')
        bw.speed = 0 # 0% duty cycle

class yellowlight(TrafficObject):
    def set_car_state(self, car_state):
        logging.debug('yellowLight: slowing car')
        bw.speed = SPEED/2
       
class stopSign(TrafficObject):
    def set_car_state(self, car_state):
        bw.speed = 0 # 0% duty cycle
        wait.done(3)
        bw.speed = SPEED

    def __init__(self, wait_time_in_sec=3, min_no_stop_sign=20):
        self.in_wait_mode = False
        self.has_stopped = False
        self.wait_time_in_sec = wait_time_in_sec
        self.min_no_stop_sign = min_no_stop_sign
        self.no_stop_count = min_no_stop_sign
        self.timer = None

    def set_car_state(self, car_state):
        self.no_stop_count = self.min_no_stop_sign

        if self.in_wait_mode:
            logging.debug('stop sign: 2) still waiting')
            # wait for 2 second before proceeding
            car_state['speed'] = 0
            return

        if not self.has_stopped:
            logging.debug('stop sign: 1) just detected')

            car_state['speed'] = 0
            self.in_wait_mode = True
            self.has_stopped = True
            self.timer = Timer(self.wait_time_in_sec, self.wait_done)
            self.timer.start()
            return

    def wait_done(self):
        logging.debug('stop sign: 3) finished waiting for %d seconds' % self.wait_time_in_sec)
        time.sleep(self)
        self.in_wait_mode = False

    def clear(self):
        if self.has_stopped:
            # need this counter in case object detection has a glitch that one frame does not
            # detect stop sign, make sure we see 20 consecutive no stop sign frames (about 1 sec)
            # and then mark has_stopped = False
            self.no_stop_count -= 1
            if self.no_stop_count == 0:
                logging.debug("stop sign: 4) no more stop sign detected")
                self.has_stopped = False
                self.in_wait_mode = False  # may not need to set this
