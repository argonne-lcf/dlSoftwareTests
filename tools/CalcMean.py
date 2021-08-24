
import math

class CalcMean:
   def __init__(self):
      self._mean = 0
      self._sigma = 0
      self._n = 0
      self._sum = 0
      self._sum2 = 0

   def add_value(self,value):
      self._n += 1
      self._sum += value
      self._sum2 += value*value
      self._mean = 0
      self._sigma = 0

   def mean(self):
      if self._mean != 0:
         return self._mean
      if self._n == 0:
         return 0

      self._mean = float(self._sum)/float(self._n)
      return self._mean

   def sigma(self):
      if self._sigma != 0:
         return self._sigma
      if self._n == 0:
         return 0
      mean = self.mean()
      self._sigma = math.sqrt( (1./self._n)*self._sum2 - mean*mean)
      return self._sigma

   def __add__(self,other):
      new = CalcMean()
      new._n = self._n + other._n
      new._sum = self._sum + other._sum
      new._sum2 = self._sum2 + other._sum2
      return new

   def __eq__(self,other):
      if self.mean() == other.mean():
         return True
      return False

   def __ne__(self,other):
      if self.mean() != other.mean():
         return True
      return False

   def __gt__(self,other):
      if self.mean() > other.mean():
         return True
      return False

   def __lt__(self,other):
      if self.mean() < other.mean():
         return True
      return False

   def __ge__(self,other):
      if self.mean() >= other.mean():
         return True
      return False

   def __le__(self,other):
      if self.mean() <= other.mean():
         return True
      return False


   def get_string(self,format = '%f +/- %f',
                       show_percent_error=False,
                       show_percent_error_format = '%12.2f +/- %12.2f (%04.2f%%)'
                 ):
      s = ''
      if show_percent_error:
         percent_error = self.sigma()/self.mean()*100.
         s = show_percent_error_format % (self.mean(),self.sigma(),percent_error)
      else:
         s = format % (self.mean(),self.sigma())
      return s

