import sys
import numpy as np

#Just checking if the user gave appropriate inputs, not gonna go through the trouble of checking if they gave number inputs though
#Also stores the arguments into the array answer_key
l = len(sys.argv)
answer_key = np.zeros((4),float)
if l == 1 :
        sys.exit("No input arguments, please try again\n")
elif l == 2 :
        print("Your function is: y = %s" % sys.argv[1])
        answer_key[3] = float(sys.argv[1])
elif l == 3 :
        print("Your function is: y = (%s)x + (%s)" % (sys.argv[1],sys.argv[2]))
        answer_key[3] = float(sys.argv[2])
        answer_key[2] = float(sys.argv[1])
elif l == 4 :
        print("Your function is: y = (%s)x^2 + (%s)x + (%s)" % (sys.argv[1],sys.argv[2],sys.argv[3]))
        answer_key[3] = float(sys.argv[3])
        answer_key[2] = float(sys.argv[2])
        answer_key[1] = float(sys.argv[1])
elif l >= 5 :
        if l>5 :
                print("Warning: Code is only designed for polynomials up to the third degree\n Extra arguments disregarded")
        print("Your function is: y = (%s)x^3 + (%s)x^2 + (%s)x + (%s)" % (sys.argv[l-4],sys.argv[l-3],sys.argv[l-2],sys.argv[l-1]))
        answer_key[3] = float(sys.argv[l-1])
        answer_key[2] = float(sys.argv[l-2])
        answer_key[1] = float(sys.argv[l-3])
        answer_key[0] = float(sys.argv[l-4])
        

#The next part creates the input and output data
datapoints = 20
distance = 100
if l == 2 :
        x = np.linspace(-distance,distance,datapoints)
elif l == 3 :
        x = np.linspace(-distance/answer_key[2],distance/answer_key[2],datapoints) 
        #range is divided by slope so that there will be less datapoints that are affected more by the noise
        #smaller slopes tend to be more affected by noise since they'll output smaller y's
elif l == 4 :
        center = -answer_key[2]/(2*answer_key[1])
        dist = distance**0.5/answer_key[1]**0.5
        x = np.linspace(center-dist+0.5,center+dist+0.5,datapoints)
else :
        center = -answer_key[1]/(3*answer_key[0])
        dist = distance**(1/3)/answer_key[0]**(1/3)
        x = np.linspace(center-dist+0.5,center+dist+0.5,datapoints)
        
input_data = np.transpose( np.array([ (x**3).tolist() , (x**2).tolist() , x.tolist() , [1.0]*datapoints ]) )
clean_output_data = input_data @ answer_key
dirty_output_data = clean_output_data + np.random.uniform(-1,1,(datapoints))
output_data = dirty_output_data

print("Generated %d datapoints and added corruption to the y data " % datapoints)
position = np.array([ 1.0 , 1.0 , 1.0 , 1.0 ])
learning_rate = 0.00001
dx = 0.0000000001
step = np.array([ 1.0 , 1.0 , 1.0 , 1.0 ])
precision = 0.00000000001
iterations = 0.0
cost_list = [];

def cost(pos):
        return np.sum( ( input_data @ pos - output_data ) ** 2 )
def cost_gradient(pos,dx):
        d = np.array([ pos.tolist() ] * 4)
        d = d + np.diag([dx]*4)
        e = [0.0]*4
        e[0] = (cost(d[0])-cost(pos))/dx
        e[1] = (cost(d[1])-cost(pos))/dx
        e[2] = (cost(d[2])-cost(pos))/dx
        e[3] = (cost(d[3])-cost(pos))/dx
        return np.array(e)

while abs(np.max(step)) > precision :
        if iterations%1000 == 0 :
            cost_list = cost_list + [cost(position)] #for diagnosis
        #cyclic_learning_rate = learning_rate*(np.sin(2*np.pi*iterations/1000)+1.01) #when I tried out cyclic learning rate
        step = -cost_gradient(position,dx)*"""cyclic_"""learning_rate
        position = position + step
        iterations = iterations + 1


print("My best guess for your function is: y = (%f)x^3 + (%f)x^2 + (%f)x + (%f)" % (position[0],position[1],position[2],position[3]) )
print("It took me %d guesses to do it!" % iterations )
print("The cost is %f" % cost(position) )
