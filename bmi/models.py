from django.db import models


#  models.MODEL : ORM(object relation mapping)관련 기능을 가지고 있다.
#  ORM : 실제 데이터를 코드로 추상화 해놓고 사용한다.
#  데이터를 저장,확인,수정,삭제
class BMI(models.Model):
    weight    = models.FloatField()
    height    = models.FloatField()
    bmi_score = models.FloatField(blank=True) #입력을 필수로 하지 않아도 된다. blank=True
    created   = models.DateTimeField(auto_now_add=True)

    def  __str__(self):
        return "키 : " + str(self.height) + "체중 :" + str(self.weight) +" BMI : "+ str(self.bmi_score) 


    #force_insert=오류가 있을때 무시함 , 이런건 알아보면 될듯 추가 옵션 같은게 있을듯
    def save(self, force_insert=False, force_update=False, using=None, update_fields=None):
        self.bmi_score = self.weight / (self.height/100)**2
        super(BMI, self).save(force_insert, force_update,using,update_fields)


