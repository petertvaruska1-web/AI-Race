// The lab glass (design §6): small square glass tiles in the main color's tint, with a bevel,
// thin grout and a slow band of light, seen through them nothing but aurora curtains of the
// second color, and a soft lamp behind the glass that follows the pointer. `uLight` blends dark
// glass (0) into light glass (1); `uPower` brightens the aurora (founding powers it up).

export const GLASS_VERTEX = "attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}";

export const GLASS_FRAGMENT = `precision highp float;
uniform vec2 uRes;uniform float uTime;uniform vec3 uP;uniform vec3 uS;uniform float uLight;
uniform float uCell;uniform vec2 uMouse;uniform float uPower;
float hash(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
 return mix(mix(hash(i),hash(i+vec2(1.,0.)),f.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),f.x),f.y);}
float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<4;i++){v+=a*noise(p);p=p*2.03+17.1;a*=.5;}return v;}
float aurora(vec2 uv,float t){float a=0.;
 for(int i=0;i<3;i++){float fi=float(i);float x=uv.x*1.6;
  float y0=0.66-0.17*fi+0.10*sin(x*1.9+t*0.06*(1.+fi)+fi*1.7)+0.05*sin(x*4.3-t*0.09+fi*2.3)+0.03*sin(x*9.+t*0.13);
  float w=0.085+0.035*sin(t*0.05+fi*2.);float d=(uv.y-y0)/w;float body=exp(-d*d);
  float rise=exp(-max(d,0.)*0.7)*smoothstep(-0.3,0.8,d)*0.5;
  float rays=0.42+0.58*fbm(vec2(x*7.+fi*5.,t*0.05+fi));a+=(body+rise)*rays*(0.78-0.18*fi);}
 return a;}
void main(){
 vec2 frag=gl_FragCoord.xy;vec2 cid=floor(frag/uCell);vec2 luv=fract(frag/uCell);
 vec2 cuv=(cid+0.5)*uCell/uRes;vec2 uv=frag/uRes;float t=uTime;
 float a=clamp(mix(aurora(cuv,t),aurora(uv,t),0.15),0.,1.6)*uPower;float h=hash(cid);
 vec3 aCol=mix(uS,mix(uS,uP,0.55),smoothstep(0.75,0.12,cuv.y));
 vec3 dCol=mix(vec3(0.012,0.014,0.024),uP*0.22,0.42)+aCol*a*0.85*(0.48+0.24*h)+uP*0.035*(1.-cuv.y);
 vec3 lBase=mix(vec3(0.958,0.962,0.97),uP,0.075);
 vec3 lCol=mix(lBase,mix(lBase,mix(uS,vec3(1.),0.22),0.62),clamp(a*0.62,0.,0.72)*(0.8+0.2*h));
 vec3 col=mix(dCol,lCol,uLight);
 vec2 e=min(luv,1.-luv);float edge=min(e.x,e.y);float grout=1.-smoothstep(0.,0.07,edge);
 float hi=(1.-smoothstep(0.,0.2,1.-luv.y))*0.6+(1.-smoothstep(0.,0.2,luv.x))*0.4;
 float lo=(1.-smoothstep(0.,0.2,luv.y))*0.6+(1.-smoothstep(0.,0.2,1.-luv.x))*0.4;
 col+=mix(vec3(0.05),vec3(0.035),uLight)*hi;col-=col*mix(0.24,0.05,uLight)*lo;
 col*=1.-grout*mix(0.4,0.08,uLight);col*=0.975+0.05*h;
 col+=pow(max(0.,sin((cuv.x*0.9+cuv.y*0.55)*3.2-t*0.12)),24.)*mix(0.04,0.03,uLight);
 vec2 asp=vec2(uRes.x/uRes.y,1.);float m=exp(-pow(length((cuv-uMouse)*asp)/0.16,2.));
 col+=m*mix(mix(uP,vec3(1.),0.35)*0.12,vec3(0.025),uLight);
 col+=(hash(frag+fract(t)*91.)-0.5)*0.014;
 float v=smoothstep(1.3,0.3,length((uv-0.5)*vec2(asp.x*0.85,1.)));
 col*=mix(mix(0.5,1.,v),mix(0.95,1.02,v),uLight);
 gl_FragColor=vec4(col,1.);
}`;
