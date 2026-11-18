"""Contact frames expressed in their parent URDF link's local coordinates."""
import numpy as np


def override_basis(record):
    # Directions refer to a frozen source frame, never to an already corrected frame.
    normal=np.asarray(record['N'],dtype=float)
    return contact_basis(normal,record['reference_N'])


def contact_basis(normal,edge):
    n=np.asarray(normal,dtype=float);n=n/np.linalg.norm(n)
    t=np.asarray(edge,dtype=float);t=t-n*np.dot(t,n)
    if np.linalg.norm(t)<1e-12:
        axis=np.eye(3)[np.argmin(np.abs(n))];t=axis-n*np.dot(axis,n)
    t=t/np.linalg.norm(t);t2=np.cross(n,t);t2/=np.linalg.norm(t2)
    return n,t,t2


def add_arrow(fig,origin,direction,length,color,label,group=None):
    import plotly.graph_objects as go
    origin=np.asarray(origin);direction=np.asarray(direction);end=origin+length*direction
    fig.add_trace(go.Scatter3d(x=[origin[0],end[0]],y=[origin[1],end[1]],z=[origin[2],end[2]],
        mode='lines+text',text=['',label],textfont=dict(color=color),line=dict(color=color,width=5),name=label,legendgroup=group,showlegend=False))
    fig.add_trace(go.Cone(x=[end[0]],y=[end[1]],z=[end[2]],u=[direction[0]],v=[direction[1]],w=[direction[2]],
        sizemode='absolute',sizeref=length*.12,anchor='tip',showscale=False,showlegend=False,
        colorscale=[[0,color],[1,color]],name=label,legendgroup=group))


def add_link_axes(fig,link):
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[0],mode='markers+text',text=['O: '+link],marker=dict(size=4,color='black'),name=link+' 原点'))
    for axis,color,direction in zip('XYZ',['darkred','darkgreen','darkblue'],np.eye(3)):
        add_arrow(fig,np.zeros(3),direction,.008,color,'+'+axis+' ('+link+')','link_frame')
