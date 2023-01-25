import uvicorn
import multiprocessing

if __name__ == "__main__":
    # multiprocessing.freeze_support() # Windows ?  "不然打包之后的exe用多进程读数据时就会弹出多个界面。"
    uvicorn.run("api:app", host='0.0.0.0', port=8000) 