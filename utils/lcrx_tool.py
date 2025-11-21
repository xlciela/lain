# define the function of subL(deltP, atr)
def myL(p_sl,p0):
    #return the subjective Leverage of  the deal
    sub_L = 1/ ((abs(p_sl-p0))/p0)
    return sub_L


#买卖决策模块：信号识别，信号过滤，信号执行




#define the opening position
class Pos(object):
    def __init__(self,capital,M=20):
        self.capital = capital
        self.M = M
    def Risk(self):
        #define the fixed risk of  individual deal
        capital = self.capital
        if capital <= 500:
            f = 0.05
        elif 500<capital<=2000:
            f = 0.03
        elif 2000<capital<= 5000:
            f = 0.02
        else:
            f = 0.02
        risk = capital* f
        return risk
    def get_pos(self,p0,p_sl):
        M = self.M
        L = myL(p_sl, p0)
        risk = self.Risk()
        opening_Pos = int(risk*L/p0)
        return opening_Pos

#动态调仓模块，引入参数ATR, 时间,逻辑是利用浮盈来扩大盈利的仓位,但是不扩大Risk
# class Order(object):
#     def __init__(self,)




#-----test----
#print(myL(53,55))
pos_Dot = Pos(100)
print(pos_Dot.get_pos(40,39))

    